# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
NEPA pretraining for Electron Microscopy (EM) images.

Two-stage training pipeline
===========================

Stage 1  –  Tiling pretraining
    Large EM images are systematically tiled into small patches
    (e.g. 224×224) and the model is pretrained with next-embedding
    prediction on these tiles.

Stage 2  –  Progressive resolution
    Resume from the Stage-1 checkpoint and progressively increase
    the tile / image resolution  (e.g. 336 → 448 → 672).
    Each resolution step is a separate run that loads the previous
    checkpoint.  RoPE position embeddings transfer seamlessly across
    resolutions because they are computed dynamically from normalised
    patch coordinates.

Examples
--------
# Stage 1: tile-based pretraining at 224×224
torchrun --nproc_per_node=4 run_nepa_em.py \\
    --image_dir ./data/em_train \\
    --val_image_dir ./data/em_val \\
    --tile_size 224 --tile_overlap 0.0 \\
    --config_name configs/pretrain/nepa-base-patch14-em \\
    --output_dir ./checkpoints/em_stage1 \\
    --per_device_train_batch_size 64 \\
    --num_train_epochs 300 --do_train --bf16

# Stage 2: continue at 448×448
torchrun --nproc_per_node=4 run_nepa_em.py \\
    --image_dir ./data/em_train \\
    --val_image_dir ./data/em_val \\
    --tile_size 448 \\
    --model_name_or_path ./checkpoints/em_stage1 \\
    --config_name configs/pretrain/nepa-base-patch14-em \\
    --output_dir ./checkpoints/em_stage2_448 \\
    --per_device_train_batch_size 16 \\
    --num_train_epochs 100 --do_train --bf16
"""

import logging
import os
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomApply,
    RandomHorizontalFlip,
    RandomResizedCrop,
    RandomVerticalFlip,
    Resize,
    ToTensor,
    GaussianBlur,
)

import transformers
from transformers import (
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.trainer_pt_utils import get_parameter_names

from models.vit_nepa import ViTNepaForPreTraining, ViTNepaConfig

logger = logging.getLogger(__name__)

# Supported image file extensions
EM_EXTENSIONS = {
    ".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp",
}


# ====================================================================
# Image loading utilities
# ====================================================================

def load_em_image(path: str) -> Image.Image:
    """Load an EM image, handling 8 / 16 / 32-bit and float depths.

    Always returns a PIL Image in ``'L'`` (grayscale) mode.
    For 16-/32-bit images the dynamic range is clipped to the
    [0.5 %, 99.5 %] percentile window and linearly mapped to [0, 255].
    """
    img = Image.open(path)

    if img.mode in ("I;16", "I;16B", "I;16L", "I"):
        arr = np.array(img, dtype=np.float32)
        lo, hi = np.percentile(arr, [0.5, 99.5])
        arr = np.clip(arr, lo, hi)
        arr = (arr - lo) / (hi - lo + 1e-8) * 255.0
        return Image.fromarray(arr.astype(np.uint8), mode="L")

    if img.mode == "F":
        arr = np.array(img, dtype=np.float32)
        lo, hi = np.percentile(arr, [0.5, 99.5])
        arr = np.clip(arr, lo, hi)
        arr = (arr - lo) / (hi - lo + 1e-8) * 255.0
        return Image.fromarray(arr.astype(np.uint8), mode="L")

    if img.mode in ("RGB", "RGBA"):
        return img.convert("L")

    if img.mode == "L":
        return img

    # Fallback
    try:
        return img.convert("L")
    except Exception:
        return img.convert("RGB").convert("L")


def scan_images(root_dir: str) -> list[str]:
    """Recursively find all image files under *root_dir*."""
    paths = []
    for p in sorted(Path(root_dir).rglob("*")):
        if p.is_file() and p.suffix.lower() in EM_EXTENSIONS:
            paths.append(str(p))
    return paths


# ====================================================================
# EM tiling dataset
# ====================================================================

class EMTilingDataset(Dataset):
    """Dataset that yields tiles extracted from large EM images.

    Two modes of operation:

    *Grid mode* (``random_crop=False``, default)
        Pre-computes a deterministic grid of non-/overlapping tile
        positions covering every source image.  Good for Stage 1
        when you want full coverage of the data.

    *Random-crop mode* (``random_crop=True``)
        Each ``__getitem__`` call picks a random image and extracts
        a random crop.  ``num_random_crops`` controls virtual epoch
        length.

    Parameters
    ----------
    image_dir : str
        Root directory containing EM images (searched recursively).
    tile_size : int
        Edge length of the square tile in pixels.
    tile_overlap : float
        Fractional overlap in [0, 1) between neighbouring grid tiles.
    random_crop : bool
        If True, use random-crop mode instead of grid mode.
    num_random_crops : int
        Virtual epoch length when ``random_crop=True``.
    transform : callable or None
        Torchvision transform applied to every tile **after** extraction.
    to_rgb : bool
        If True, convert grayscale tiles to 3-channel RGB
        (required when ``config.num_channels == 3``).
    """

    def __init__(
        self,
        image_dir: str,
        tile_size: int = 224,
        tile_overlap: float = 0.0,
        random_crop: bool = False,
        num_random_crops: int = 100_000,
        transform=None,
        to_rgb: bool = True,
    ):
        super().__init__()
        self.tile_size = tile_size
        self.random_crop = random_crop
        self.num_random_crops = num_random_crops
        self.transform = transform
        self.to_rgb = to_rgb

        self.image_paths = scan_images(image_dir)
        if not self.image_paths:
            raise FileNotFoundError(f"No images found in {image_dir}")
        logger.info(f"Found {len(self.image_paths)} images in {image_dir}")

        # --- Grid mode: pre-compute tile positions ---
        self.tiles: list[tuple[int, int, int, bool]] = []
        if not random_crop:
            stride = max(1, int(tile_size * (1.0 - tile_overlap)))
            for img_idx, path in enumerate(self.image_paths):
                try:
                    with Image.open(path) as img:
                        w, h = img.size
                except Exception as exc:
                    logger.warning(f"Skipping {path}: {exc}")
                    continue

                if h < tile_size or w < tile_size:
                    # Smaller than one tile → resize later
                    self.tiles.append((img_idx, 0, 0, True))
                    continue

                for y in range(0, h - tile_size + 1, stride):
                    for x in range(0, w - tile_size + 1, stride):
                        self.tiles.append((img_idx, y, x, False))

            logger.info(
                f"Grid tiling: {len(self.tiles)} tiles  "
                f"(tile={tile_size}, overlap={tile_overlap})"
            )

    # ----------------------------------------------------------------

    def __len__(self) -> int:
        if self.random_crop:
            return self.num_random_crops
        return len(self.tiles)

    def _to_rgb_if_needed(self, img: Image.Image) -> Image.Image:
        if self.to_rgb and img.mode == "L":
            return img.convert("RGB")
        return img

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if self.random_crop:
            return self._getitem_random(idx)
        return self._getitem_grid(idx)

    # --- random-crop mode ---
    def _getitem_random(self, _idx: int) -> dict[str, torch.Tensor]:
        img_idx = random.randint(0, len(self.image_paths) - 1)
        img = load_em_image(self.image_paths[img_idx])

        w, h = img.size
        ts = self.tile_size

        # Up-scale tiny images so we can crop
        if h < ts or w < ts:
            scale = max(ts / h, ts / w) + 0.01
            img = img.resize(
                (int(w * scale), int(h * scale)), Image.BICUBIC
            )
            w, h = img.size

        y = random.randint(0, h - ts)
        x = random.randint(0, w - ts)
        tile = img.crop((x, y, x + ts, y + ts))

        tile = self._to_rgb_if_needed(tile)
        if self.transform is not None:
            tile = self.transform(tile)
        return {"pixel_values": tile}

    # --- grid mode ---
    def _getitem_grid(self, idx: int) -> dict[str, torch.Tensor]:
        img_idx, y, x, needs_resize = self.tiles[idx]
        img = load_em_image(self.image_paths[img_idx])

        ts = self.tile_size
        if needs_resize:
            tile = img.resize((ts, ts), Image.BICUBIC)
        else:
            tile = img.crop((x, y, x + ts, y + ts))

        tile = self._to_rgb_if_needed(tile)
        if self.transform is not None:
            tile = self.transform(tile)
        return {"pixel_values": tile}


# ====================================================================
# Pre-tiled dataset (fast — each file is one tile, no big-image I/O)
# ====================================================================

class PretiledDataset(Dataset):
    """Ultra-fast dataset that reads pre-cut tile JPEGs.

    Use after running ``pretile_em_data.py`` which converts large
    micrographs into individual tile files.  Each ``__getitem__``
    opens a single small JPEG (~50 Kpx) instead of decoding a full
    micrograph (~56 Mpx), giving ~1000× I/O speed-up.

    Parameters
    ----------
    tile_dir : str
        Directory containing tile JPEG files.
    transform : callable or None
        Torchvision transform applied to each tile.
    to_rgb : bool
        Convert grayscale to 3-channel RGB.
    """

    def __init__(self, tile_dir: str, transform=None, to_rgb: bool = True):
        super().__init__()
        self.transform = transform
        self.to_rgb = to_rgb

        self.tile_paths = sorted([
            str(p) for p in Path(tile_dir).iterdir()
            if p.is_file() and p.suffix.lower() in EM_EXTENSIONS
        ])
        if not self.tile_paths:
            raise FileNotFoundError(f"No tile images found in {tile_dir}")
        logger.info(f"PretiledDataset: {len(self.tile_paths)} tiles from {tile_dir}")

    def __len__(self) -> int:
        return len(self.tile_paths)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        img = Image.open(self.tile_paths[idx])

        # Handle non-standard modes (shouldn't happen for pre-tiled JPEGs, but be safe)
        if img.mode not in ("L", "RGB"):
            img = img.convert("L")

        if self.to_rgb and img.mode == "L":
            img = img.convert("RGB")

        if self.transform is not None:
            img = self.transform(img)
        return {"pixel_values": img}


# ====================================================================
# EM-specific augmentations
# ====================================================================

class RandomRotate90:
    """Randomly rotate by 0 / 90 / 180 / 270 degrees (lossless)."""

    def __call__(self, img: Image.Image) -> Image.Image:
        k = random.randint(0, 3)
        if k == 0:
            return img
        return img.rotate(k * 90, expand=False)


class AddGaussianNoise:
    """Add Gaussian noise to a tensor (applied *after* ToTensor)."""

    def __init__(self, max_std: float = 0.03):
        self.max_std = max_std

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        std = random.uniform(0.0, self.max_std)
        if std > 0:
            tensor = tensor + torch.randn_like(tensor) * std
        return tensor


def build_train_transforms(
    tile_size: int,
    extra_random_crop: bool = False,
) -> Compose:
    """Training transforms for EM tiles.

    - Random flips (horizontal + vertical)
    - Random 90° rotations
    - Optional Gaussian blur
    - Gaussian noise injection
    - Normalize to [-1, 1]
    """
    ops: list = []

    if extra_random_crop:
        # Additional random sub-crop for more spatial jitter
        ops.append(RandomResizedCrop(tile_size, scale=(0.6, 1.0), interpolation=Image.BICUBIC))

    ops += [
        RandomHorizontalFlip(p=0.5),
        RandomVerticalFlip(p=0.5),
        RandomRotate90(),
        RandomApply([GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))], p=0.2),
        ToTensor(),
        AddGaussianNoise(max_std=0.03),
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
    return Compose(ops)


def build_val_transforms(tile_size: int) -> Compose:
    """Deterministic validation transforms for EM images."""
    return Compose([
        Resize(tile_size, interpolation=Image.BICUBIC),
        CenterCrop(tile_size),
        ToTensor(),
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


# ====================================================================
# Trainer with EMA + optional embedding LR
# ====================================================================

class EMTrainer(Trainer):
    """Extends HF Trainer with EMA and per-group learning rates."""

    def __init__(
        self,
        *args,
        embed_lr: float | None = None,
        ema_decay: float = 0.9999,
        use_ema: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.embed_lr = embed_lr
        self.ema_decay = ema_decay
        self.use_ema = use_ema
        self.ema_model = None

    # -- weight-decay filter -------------------------------------------

    def get_decay_parameter_names(self, model) -> list[str]:
        forbidden = [
            r"bias", r"layernorm", r"rmsnorm", r"layer_scale",
            r"(?:^|\.)norm(?:$|\.)", r"_norm(?:$|\.)",
        ]
        return get_parameter_names(model, [torch.nn.LayerNorm], forbidden)

    # -- separate embedding LR -----------------------------------------

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer
        if self.embed_lr is None:
            return super().create_optimizer()

        decay_names = set(self.get_decay_parameter_names(self.model))
        embed_names = {
            f"vit_nepa.embeddings.{n}"
            for n, _ in self.model.vit_nepa.embeddings.named_parameters()
        }

        wd = self.args.weight_decay
        base_lr = self.args.learning_rate

        groups = [
            {"params": [], "weight_decay": wd,  "lr": self.embed_lr},   # embed + decay
            {"params": [], "weight_decay": 0.0, "lr": self.embed_lr},   # embed + no decay
            {"params": [], "weight_decay": wd,  "lr": base_lr},         # body  + decay
            {"params": [], "weight_decay": 0.0, "lr": base_lr},         # body  + no decay
        ]

        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            is_embed = name in embed_names
            is_decay = name in decay_names
            idx = (0 if is_embed else 2) + (0 if is_decay else 1)
            groups[idx]["params"].append(p)

        groups = [g for g in groups if g["params"]]
        opt_cls, opt_kw = self.get_optimizer_cls_and_kwargs(self.args)
        self.optimizer = opt_cls(groups, **opt_kw)
        return self.optimizer

    # -- EMA ------------------------------------------------------------

    def _init_ema(self):
        import copy
        self.ema_model = copy.deepcopy(self.model).eval().float()
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

    def _update_ema(self):
        if not self.use_ema:
            return
        if self.ema_model is None:
            self._init_ema()
        with torch.no_grad():
            msd = self.model.state_dict()
            for k, v in self.ema_model.state_dict().items():
                if k in msd:
                    v.mul_(self.ema_decay).add_(msd[k].float(), alpha=1.0 - self.ema_decay)

    def _maybe_log_save_evaluate(self, *args, **kwargs):
        if self.state.global_step > getattr(self, "_ema_step", 0):
            self._update_ema()
            self._ema_step = self.state.global_step
        super()._maybe_log_save_evaluate(*args, **kwargs)

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval", **kw):
        out = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix, **kw)
        if self.use_ema and self.ema_model is not None:
            backup = self.model
            self.model = self.ema_model
            ema_out = super().evaluate(
                eval_dataset, ignore_keys, metric_key_prefix + "_ema", **kw
            )
            self.model = backup
            out.update(ema_out)
        return out

    def predict(self, test_dataset, ignore_keys=None, metric_key_prefix="test", **kw):
        out = super().predict(test_dataset, ignore_keys, metric_key_prefix, **kw)
        if self.use_ema and self.ema_model is not None:
            backup = self.model
            self.model = self.ema_model
            ema_out = super().predict(
                test_dataset, ignore_keys, metric_key_prefix + "_ema", **kw
            )
            self.model = backup
            from transformers.trainer_utils import PredictionOutput
            if isinstance(out, PredictionOutput) and isinstance(ema_out, PredictionOutput):
                out.metrics.update(ema_out.metrics)
        return out

    # -- eval loss fix (NEPA has no labels, but loss is intrinsic) --------

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """Override to always capture loss.

        The default Trainer skips loss collection when there are no labels.
        NEPA computes loss internally from next-embedding prediction, so we
        always extract it from the model output.
        """
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            with self.compute_loss_context_manager():
                outputs = model(**inputs)
            loss = outputs.loss
            if loss is not None:
                loss = loss.mean().detach()
        return (loss, None, None)

    # -- checkpoint save / load with EMA --------------------------------

    def save_model(self, output_dir=None, _internal_call=False):
        super().save_model(output_dir, _internal_call)
        output_dir = output_dir or self.args.output_dir
        if self.use_ema and self.ema_model is not None and self.args.should_save:
            ema_path = os.path.join(output_dir, "pytorch_model_ema.bin")
            os.makedirs(os.path.dirname(ema_path), exist_ok=True)
            torch.save(self.ema_model.state_dict(), ema_path)
            self.log({"ema_saved": ema_path})

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        super()._load_from_checkpoint(resume_from_checkpoint, model)
        if self.use_ema:
            ema_ckpt = os.path.join(resume_from_checkpoint, "pytorch_model_ema.bin")
            if os.path.exists(ema_ckpt):
                if self.ema_model is None:
                    self._init_ema()
                sd = torch.load(ema_ckpt, map_location="cpu")
                missing, unexpected = self.ema_model.load_state_dict(sd, strict=False)
                if missing or unexpected:
                    logger.warning(
                        f"[EMA] load_state_dict: {len(missing)} missing, "
                        f"{len(unexpected)} unexpected"
                    )
            else:
                logger.info("[EMA] No ema checkpoint found, starting fresh.")


# ====================================================================
# Embedding visualisation callback
# ====================================================================

class EmbeddingVisCallback(transformers.TrainerCallback):
    """Periodically extract embeddings from val tiles and save UMAP/t-SNE plots.

    Visualisation is saved to ``{output_dir}/vis/embeddings_step{N}.png``.
    Tiles are coloured by their source dataset (EMPIAR ID parsed from filename).

    Parameters
    ----------
    val_dataset : Dataset
        Validation dataset (PretiledDataset or EMTilingDataset).
    n_samples : int
        Number of tiles to sample for the plot.
    every_steps : int
        Run visualisation every N training steps.
    """

    def __init__(
        self,
        val_dataset,
        n_samples: int = 2000,
        every_steps: int = 2000,
        enable_attention_maps: bool = True,
        attention_n_samples: int = 8,
        attention_last_n_layers: int = 4,
        attention_batch_size: int = 1,
    ):
        super().__init__()
        self.val_dataset = val_dataset
        self.n_samples = n_samples
        self.every_steps = every_steps
        self.enable_attention_maps = enable_attention_maps
        self.attention_n_samples = attention_n_samples
        self.attention_last_n_layers = attention_last_n_layers
        self.attention_batch_size = attention_batch_size

    @staticmethod
    def _sample_indices(dataset_len: int, n: int, seed: int) -> list[int]:
        if dataset_len <= 0 or n <= 0:
            return []
        n = min(n, dataset_len)
        rng = random.Random(seed)
        return rng.sample(range(dataset_len), n)

    def _get_dataset_ids(self) -> list[str]:
        """Try to extract dataset ID from tile filenames (e.g. '10005_xxx.jpg' → '10005')."""
        ids = []
        ds = self.val_dataset
        if hasattr(ds, "tile_paths"):
            paths = ds.tile_paths
        elif hasattr(ds, "image_paths"):
            paths = ds.image_paths
        else:
            return ["unknown"] * len(ds)

        for p in paths:
            name = os.path.basename(p)
            # Expect format: {dataset_id}_{rest}.jpg
            parts = name.split("_", 1)
            ids.append(parts[0] if parts[0].isdigit() else "unknown")
        return ids

    @torch.no_grad()
    def _extract_embeddings(self, model, device, indices: list[int]) -> tuple[np.ndarray, list[str]]:
        """Run a subset of val tiles through the model, return (embeddings, dataset_ids)."""
        import torch.nn.functional as F

        dataset_ids = self._get_dataset_ids()
        if not indices:
            return np.empty((0, 0), dtype=np.float32), []

        embeddings = []
        labels = []
        model.eval()

        batch_size = 64
        for start in range(0, len(indices), batch_size):
            batch_idx = indices[start : start + batch_size]
            pixels = torch.stack([
                self.val_dataset[i]["pixel_values"] for i in batch_idx
            ]).to(device)

            # Forward through the base ViT model (not the pretraining head)
            vit = model.vit_nepa if hasattr(model, "vit_nepa") else model
            out = vit(pixel_values=pixels)
            hidden = out.last_hidden_state  # [B, seq_len, D]

            # Mean-pool patch tokens (exclude CLS at position 0)
            patch_emb = hidden[:, 1:, :].mean(dim=1)  # [B, D]
            patch_emb = F.normalize(patch_emb, dim=-1)
            embeddings.append(patch_emb.cpu())

            for i in batch_idx:
                labels.append(dataset_ids[i] if i < len(dataset_ids) else "unknown")

        embeddings = torch.cat(embeddings, dim=0).numpy()
        return embeddings, labels

    @staticmethod
    def _to_vis_image(pixel_values: torch.Tensor) -> np.ndarray:
        """Convert normalized tensor (-1..1) to display-ready RGB numpy image (0..1)."""
        img = pixel_values.detach().cpu().float().permute(1, 2, 0).numpy()
        img = np.clip(img * 0.5 + 0.5, 0.0, 1.0)
        if img.shape[-1] == 1:
            img = np.repeat(img, 3, axis=-1)
        return img

    @staticmethod
    def _compute_query_attention_grid(
        attentions: tuple[torch.Tensor, ...],
        query_token: int,
        patch_start: int,
        grid_h: int,
        grid_w: int,
        last_n_layers: int,
    ) -> torch.Tensor:
        """NEPA-style attention map for a selected query token.

        Average selected layers and heads, then take attention row for the
        chosen query token over spatial patch tokens.
        """
        n_layers = len(attentions)
        n = max(1, min(last_n_layers, n_layers))
        selected = attentions[-n:]  # tuple([B, H, T, T], ...)
        att = torch.stack(selected, dim=0).mean(dim=0)  # [B, H, T, T]
        att = att.mean(dim=1)  # [B, T, T]
        att_vec = att[:, query_token, patch_start:]  # [B, P]
        return att_vec.view(att_vec.shape[0], grid_h, grid_w)

    @staticmethod
    def _compute_query_prob_grid(
        outputs,
        query_token: int,
        patch_start: int,
        grid_h: int,
        grid_w: int,
    ) -> torch.Tensor:
        """NEPA hidden-input probability grid for selected query token."""
        import torch.nn.functional as F

        h_pred = outputs.last_hidden_state  # [B, T, D]
        e_in = outputs.input_embedding      # [B, T, D]
        h = h_pred[:, query_token, :]       # [B, D]

        e = F.normalize(e_in[:, patch_start:, :], dim=-1)      # [B, P, D]
        h = F.normalize(h.unsqueeze(1), dim=-1)                # [B, 1, D]
        sim = (e * h).sum(dim=-1)                              # [B, P]
        prob = torch.softmax(sim, dim=-1)                      # [B, P]
        return prob.view(prob.shape[0], grid_h, grid_w)

    @staticmethod
    def _normalize_map(arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr, dtype=np.float32)
        arr = arr - arr.min()
        den = arr.max()
        if den > 1e-8:
            arr = arr / den
        return arr

    @torch.no_grad()
    def _visualize_attention_and_log(
        self,
        model,
        device,
        indices: list[int],
        step: int,
        output_dir: str,
    ) -> None:
        if not indices:
            return

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
        except ImportError:
            logger.warning("matplotlib not installed, skipping NEPA token maps.")
            return

        dataset_ids = self._get_dataset_ids()
        vit = model.vit_nepa if hasattr(model, "vit_nepa") else model
        patch_size = int(vit.config.patch_size)

        records = []
        batch_size = max(1, self.attention_batch_size)
        for start in range(0, len(indices), batch_size):
            batch_idx = indices[start : start + batch_size]
            pixels = torch.stack([
                self.val_dataset[i]["pixel_values"] for i in batch_idx
            ]).to(device)

            out = vit(pixel_values=pixels, output_attentions=True)
            if out.attentions is None or len(out.attentions) == 0:
                logger.warning("No attention tensors returned; skipping attention maps.")
                return

            for k, ds_idx in enumerate(batch_idx):
                img = self._to_vis_image(pixels[k])
                h_img, w_img = pixels[k].shape[-2], pixels[k].shape[-1]
                grid_h, grid_w = h_img // patch_size, w_img // patch_size
                num_patches = grid_h * grid_w
                patch_start = out.attentions[0].shape[-1] - num_patches

                # Match NEPA run_visualization: pick one random query patch per sample.
                rng = random.Random(step * 1_000_003 + int(ds_idx))
                q_r = rng.randint(0, grid_h - 1)
                q_c = rng.randint(0, grid_w - 1)
                query_token = patch_start + q_r * grid_w + q_c

                att_grid = self._compute_query_attention_grid(
                    attentions=out.attentions,
                    query_token=query_token,
                    patch_start=patch_start,
                    grid_h=grid_h,
                    grid_w=grid_w,
                    last_n_layers=self.attention_last_n_layers,
                )[k].detach().cpu().numpy()

                prob_grid = self._compute_query_prob_grid(
                    outputs=out,
                    query_token=query_token,
                    patch_start=patch_start,
                    grid_h=grid_h,
                    grid_w=grid_w,
                )[k].detach().cpu().numpy()

                label = dataset_ids[ds_idx] if ds_idx < len(dataset_ids) else "unknown"
                records.append((img, att_grid, prob_grid, label, ds_idx, q_r, q_c, grid_h, grid_w))

        if not records:
            return

        # One row per sample: [image | query patch | attention map | prob map]
        n_rows = len(records)
        fig, axes = plt.subplots(n_rows, 4, figsize=(18, 4.2 * n_rows))
        if n_rows == 1:
            axes = np.array([axes])

        for row, (img, att_grid, prob_grid, label, ds_idx, q_r, q_c, grid_h, grid_w) in enumerate(records):
            h_img, w_img = img.shape[0], img.shape[1]
            cell_w = w_img / grid_w
            cell_h = h_img / grid_h

            ax0, ax1, ax2, ax3 = axes[row]

            # Original
            ax0.imshow(img, cmap="gray")
            ax0.set_title(f"id={label} idx={ds_idx}", fontsize=9)
            ax0.axis("off")

            # Query patch box
            ax1.imshow(img, cmap="gray")
            rect = patches.Rectangle(
                (q_c * cell_w, q_r * cell_h),
                cell_w, cell_h,
                linewidth=2.0,
                edgecolor="red",
                facecolor="none",
            )
            ax1.add_patch(rect)
            ax1.set_title(f"Query patch (r={q_r}, c={q_c})", fontsize=9)
            ax1.axis("off")

            # Attention map for query token
            att_vis = np.power(self._normalize_map(att_grid), 0.5)
            ax2.imshow(img, cmap="gray")
            ax2.imshow(
                att_vis,
                cmap="inferno",
                alpha=0.55,
                extent=(0, w_img, h_img, 0),
                interpolation="nearest",
            )
            ax2.set_title("Attention map", fontsize=9)
            ax2.axis("off")

            # Hidden-input probability map (NEPA)
            prob_vis = np.power(self._normalize_map(prob_grid), 0.5)
            ax3.imshow(img, cmap="gray")
            ax3.imshow(
                prob_vis,
                cmap="viridis",
                alpha=0.55,
                extent=(0, w_img, h_img, 0),
                interpolation="nearest",
            )
            ax3.set_title("Embedding probability map", fontsize=9)
            ax3.axis("off")

        fig.suptitle(f"Val NEPA Token Maps — step {step}", fontsize=13, y=1.01)
        fig.tight_layout()

        vis_dir = os.path.join(output_dir, "vis")
        os.makedirs(vis_dir, exist_ok=True)
        attn_path = os.path.join(vis_dir, f"nepa_maps_step{step}.png")
        fig.savefig(attn_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved NEPA token maps → {attn_path}")

        try:
            import wandb
            if wandb.run is not None:
                wandb.log(
                    {
                        "val/nepa_maps": wandb.Image(attn_path),
                        "val/attention_maps": wandb.Image(attn_path),
                    },
                    step=step,
                )
                logger.info("Logged NEPA token maps to wandb")
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"Failed to log NEPA token maps to wandb: {e}")

    @staticmethod
    def _make_scatter(ax, coords_2d, labels, unique_labels, cmap, label_to_idx, title):
        """Draw one scatter subplot."""
        for label in unique_labels:
            mask = [i for i, l in enumerate(labels) if l == label]
            ax.scatter(
                coords_2d[mask, 0], coords_2d[mask, 1],
                c=[cmap(label_to_idx[label])],
                label=label, s=6, alpha=0.6,
            )
        ax.set_title(title, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])

    def _visualize_and_log(self, embeddings, labels, step: int, output_dir: str):
        """Run UMAP + t-SNE, save side-by-side plot, and log to wandb."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not installed, skipping embedding visualisation.")
            return

        # ---- PCA pre-reduction (speed up UMAP / t-SNE) ----
        emb = embeddings
        try:
            from sklearn.decomposition import PCA
            if emb.shape[1] > 50:
                emb = PCA(n_components=50, random_state=0).fit_transform(emb)
        except ImportError:
            pass

        # ---- Compute both projections ----
        projections: dict[str, np.ndarray] = {}

        try:
            from umap import UMAP
            projections["UMAP"] = UMAP(
                n_components=2, random_state=0, n_neighbors=15, min_dist=0.1,
            ).fit_transform(emb)
        except ImportError:
            logger.info("umap-learn not installed, skipping UMAP.")

        try:
            from sklearn.manifold import TSNE
            projections["t-SNE"] = TSNE(
                n_components=2, random_state=0, perplexity=30,
            ).fit_transform(emb)
        except ImportError:
            logger.info("sklearn not installed, skipping t-SNE.")

        if not projections:
            logger.warning("No dimensionality reduction available, skipping plot.")
            return

        # ---- Shared colour mapping ----
        unique_labels = sorted(set(labels))
        cmap = plt.cm.get_cmap("tab20", max(len(unique_labels), 1))
        label_to_idx = {l: i for i, l in enumerate(unique_labels)}

        n_plots = len(projections)
        fig, axes = plt.subplots(1, n_plots, figsize=(10 * n_plots, 8))
        if n_plots == 1:
            axes = [axes]

        for ax, (method, coords) in zip(axes, projections.items()):
            self._make_scatter(
                ax, coords, labels, unique_labels, cmap, label_to_idx,
                title=f"Val Embeddings — {method} — step {step}",
            )

        # Shared legend on the rightmost subplot
        axes[-1].legend(
            loc="center left", bbox_to_anchor=(1.02, 0.5),
            markerscale=3, fontsize=7, ncol=1,
        )

        fig.suptitle(f"Step {step}", fontsize=13, y=1.01)
        fig.tight_layout()

        # ---- Save locally ----
        vis_dir = os.path.join(output_dir, "vis")
        os.makedirs(vis_dir, exist_ok=True)
        combined_path = os.path.join(vis_dir, f"embeddings_step{step}.png")
        fig.savefig(combined_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved embedding plot → {combined_path}")

        # Save individual projection PNGs
        individual_paths: dict[str, str] = {}
        for method, coords in projections.items():
            fig_single, ax_single = plt.subplots(1, 1, figsize=(10, 8))
            self._make_scatter(
                ax_single, coords, labels, unique_labels,
                cmap, label_to_idx,
                title=f"{method} — step {step}",
            )
            ax_single.legend(
                loc="center left", bbox_to_anchor=(1.02, 0.5),
                markerscale=3, fontsize=7, ncol=1,
            )
            fig_single.tight_layout()
            safe_name = method.lower().replace("-", "_")
            single_path = os.path.join(vis_dir, f"{safe_name}_step{step}.png")
            fig_single.savefig(single_path, dpi=150, bbox_inches="tight")
            plt.close(fig_single)
            individual_paths[safe_name] = single_path

        # ---- Log to wandb (use saved file paths, not fig objects) ----
        try:
            import wandb
            if wandb.run is not None:
                log_dict = {"val/embedding_plot": wandb.Image(combined_path)}
                for safe_name, path in individual_paths.items():
                    log_dict[f"val/embedding_{safe_name}"] = wandb.Image(path)
                wandb.log(log_dict, step=step)
                logger.info(f"Logged {len(log_dict)} images to wandb (step {step})")
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"Failed to log to wandb: {e}")

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step % self.every_steps != 0 or state.global_step == 0:
            return
        if model is None:
            return
        # Only run on main process
        if args.local_rank not in (-1, 0):
            return

        logger.info(f"[EmbeddingVis] Extracting embeddings at step {state.global_step}...")
        device = next(model.parameters()).device
        was_training = model.training
        model.eval()
        try:
            emb_indices = self._sample_indices(
                dataset_len=len(self.val_dataset),
                n=self.n_samples,
                seed=0,
            )
            embeddings, labels = self._extract_embeddings(model, device, emb_indices)
            self._visualize_and_log(embeddings, labels, state.global_step, args.output_dir)

            if self.enable_attention_maps:
                attn_indices = emb_indices[: self.attention_n_samples]
                self._visualize_attention_and_log(
                    model=model,
                    device=device,
                    indices=attn_indices,
                    step=state.global_step,
                    output_dir=args.output_dir,
                )
        finally:
            if was_training:
                model.train()


# ====================================================================
# CLI argument dataclasses
# ====================================================================

@dataclass
class DataTrainingArguments:
    """Arguments for the EM data pipeline."""

    image_dir: str = field(
        metadata={"help": "Root directory containing training EM images (scanned recursively)."},
    )
    val_image_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Root directory containing validation EM images."},
    )
    tile_size: int = field(
        default=224,
        metadata={
            "help": (
                "Tile edge length in pixels.  Must be divisible by patch_size (14). "
                "Stage 1 typically uses 224; Stage 2 uses 336 / 448 / 672."
            )
        },
    )
    tile_overlap: float = field(
        default=0.0,
        metadata={"help": "Overlap ratio between adjacent tiles (0.0 – 0.9).  0.5 = 50 %% overlap."},
    )
    random_crop: bool = field(
        default=False,
        metadata={
            "help": (
                "Use random cropping instead of deterministic grid tiling. "
                "More diverse but does not guarantee full image coverage."
            )
        },
    )
    num_random_crops: int = field(
        default=100_000,
        metadata={"help": "Virtual epoch length when random_crop is True."},
    )
    extra_random_crop: bool = field(
        default=False,
        metadata={"help": "Apply an additional RandomResizedCrop on top of each tile for extra jitter."},
    )
    pretiled: bool = field(
        default=False,
        metadata={
            "help": (
                "If True, image_dir / val_image_dir contain pre-cut tile JPEGs "
                "(from pretile_em_data.py).  ~1000× faster data loading — no "
                "large-image decoding at training time."
            )
        },
    )
    visualize_embeddings: bool = field(
        default=False,
        metadata={"help": "Save UMAP/t-SNE of val embeddings during training (coloured by dataset)."},
    )
    vis_every_steps: int = field(
        default=2000,
        metadata={"help": "How often (in steps) to generate embedding visualisation."},
    )
    vis_n_samples: int = field(
        default=2000,
        metadata={"help": "Number of val tiles to use for embedding visualisation."},
    )
    vis_attention_maps: bool = field(
        default=True,
        metadata={"help": "Also generate NEPA token maps (query patch + attention + prob) during visualisation."},
    )
    vis_attention_n_samples: int = field(
        default=8,
        metadata={"help": "Number of val tiles for NEPA token-map visualisation per step."},
    )
    vis_attention_last_n_layers: int = field(
        default=4,
        metadata={"help": "Average attention over the last N transformer layers for NEPA attention map."},
    )
    vis_attention_batch_size: int = field(
        default=1,
        metadata={"help": "Batch size for attention-map extraction."},
    )


@dataclass
class ModelArguments:
    """Arguments for model initialisation."""

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained model checkpoint (local dir or HF hub id).  None = train from scratch."},
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={"help": "Path to model config.  Falls back to model_name_or_path."},
    )
    cache_dir: Optional[str] = field(default=None)
    model_revision: str = field(default="main")
    token: Optional[str] = field(default=None)
    trust_remote_code: bool = field(default=False)
    embed_lr: Optional[float] = field(
        default=None,
        metadata={"help": "Separate learning rate for the patch-embedding layer."},
    )
    use_ema: bool = field(
        default=True,
        metadata={"help": "Maintain an Exponential Moving Average of model weights."},
    )
    ema_decay: float = field(
        default=0.9999,
        metadata={"help": "EMA decay coefficient."},
    )
    pos_embed_rescale: Optional[float] = field(
        default=None,
        metadata={"help": "Override config.pos_embed_rescale (larger → better resolution transfer)."},
    )


# ====================================================================
# Collator
# ====================================================================

def collate_fn(examples: list[dict]) -> dict[str, torch.Tensor]:
    pixel_values = torch.stack([ex["pixel_values"] for ex in examples])
    return {"pixel_values": pixel_values}


# ====================================================================
# Main
# ====================================================================

def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # ---- Logging ----
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, "
        f"n_gpu: {training_args.n_gpu}, "
        f"distributed: {training_args.parallel_mode.value == 'distributed'}, "
        f"fp16: {training_args.fp16}, bf16: {training_args.bf16}"
    )
    logger.info(f"Training args: {training_args}")
    logger.info(f"Data args:     {data_args}")
    logger.info(f"Model args:    {model_args}")

    # ---- Resume detection ----
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not "
                "empty.  Use --overwrite_output_dir to override."
            )
        if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(f"Checkpoint detected – resuming from {last_checkpoint}")

    set_seed(training_args.seed)

    # ---- Validation ----
    patch_size = 14  # will be confirmed from config below
    assert data_args.tile_size >= patch_size, (
        f"tile_size ({data_args.tile_size}) must be >= patch_size"
    )

    # ---- Config ----
    config_path = model_args.config_name or model_args.model_name_or_path
    if config_path:
        config = ViTNepaConfig.from_pretrained(
            config_path,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
        )
    else:
        logger.info("No config path given – using default ViTNepaConfig.")
        config = ViTNepaConfig()

    patch_size = config.patch_size
    assert data_args.tile_size % patch_size == 0, (
        f"tile_size ({data_args.tile_size}) must be divisible by "
        f"patch_size ({patch_size}).  "
        f"Nearest valid sizes: {data_args.tile_size - data_args.tile_size % patch_size}, "
        f"{data_args.tile_size - data_args.tile_size % patch_size + patch_size}"
    )

    # Override image_size to match the current tile_size
    config.image_size = data_args.tile_size

    if model_args.pos_embed_rescale is not None:
        config.pos_embed_rescale = model_args.pos_embed_rescale

    num_patches_per_side = data_args.tile_size // patch_size
    seq_len = num_patches_per_side ** 2
    logger.info(
        f"Resolution: {data_args.tile_size}×{data_args.tile_size}  →  "
        f"{num_patches_per_side}×{num_patches_per_side} = {seq_len} patches"
    )

    # ---- Model ----
    if model_args.model_name_or_path:
        model = ViTNepaForPreTraining.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
            ignore_mismatched_sizes=True,  # safe: no size-dependent params
        )
        logger.info(f"Loaded pretrained weights from {model_args.model_name_or_path}")
    else:
        model = ViTNepaForPreTraining(config)
        logger.info("Training new model from scratch.")

    # ---- Transforms ----
    train_tf = build_train_transforms(
        tile_size=data_args.tile_size,
        extra_random_crop=data_args.extra_random_crop,
    )
    val_tf = build_val_transforms(tile_size=data_args.tile_size)

    # ---- Datasets ----
    is_rgb = (config.num_channels == 3)

    if data_args.pretiled:
        # Fast path: each file is one tile, no large-image decoding
        logger.info("Using PretiledDataset (fast mode)")
        train_dataset = PretiledDataset(
            tile_dir=data_args.image_dir,
            transform=train_tf,
            to_rgb=is_rgb,
        )
        val_dataset = None
        if data_args.val_image_dir is not None:
            val_dataset = PretiledDataset(
                tile_dir=data_args.val_image_dir,
                transform=val_tf,
                to_rgb=is_rgb,
            )
    else:
        # Standard path: tile large images on-the-fly
        train_dataset = EMTilingDataset(
            image_dir=data_args.image_dir,
            tile_size=data_args.tile_size,
            tile_overlap=data_args.tile_overlap,
            random_crop=data_args.random_crop,
            num_random_crops=data_args.num_random_crops,
            transform=train_tf,
            to_rgb=is_rgb,
        )
        val_dataset = None
        if data_args.val_image_dir is not None:
            val_dataset = EMTilingDataset(
                image_dir=data_args.val_image_dir,
                tile_size=data_args.tile_size,
                tile_overlap=0.0,   # no overlap for deterministic val
                random_crop=False,
                transform=val_tf,
                to_rgb=is_rgb,
            )

    logger.info(f"Train samples: {len(train_dataset)}")
    if val_dataset is not None:
        logger.info(f"Val   samples: {len(val_dataset)}")

    # ---- Callbacks ----
    callbacks = []
    if data_args.visualize_embeddings and val_dataset is not None:
        callbacks.append(
            EmbeddingVisCallback(
                val_dataset=val_dataset,
                n_samples=data_args.vis_n_samples,
                every_steps=data_args.vis_every_steps,
                enable_attention_maps=data_args.vis_attention_maps,
                attention_n_samples=data_args.vis_attention_n_samples,
                attention_last_n_layers=data_args.vis_attention_last_n_layers,
                attention_batch_size=data_args.vis_attention_batch_size,
            )
        )
        logger.info(
            f"Embedding visualisation enabled: every {data_args.vis_every_steps} steps, "
            f"{data_args.vis_n_samples} samples, nepa_maps={data_args.vis_attention_maps} "
            f"(n={data_args.vis_attention_n_samples}, last_layers={data_args.vis_attention_last_n_layers}, "
            f"attn_bs={data_args.vis_attention_batch_size}) "
            f"→ {{output_dir}}/vis/"
        )

    # ---- Trainer ----
    trainer = EMTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=val_dataset if training_args.do_eval else None,
        data_collator=collate_fn,
        embed_lr=model_args.embed_lr,
        use_ema=model_args.use_ema,
        ema_decay=model_args.ema_decay,
        callbacks=callbacks,
    )

    # ---- Train ----
    if training_args.do_train:
        ckpt = training_args.resume_from_checkpoint or last_checkpoint
        result = trainer.train(resume_from_checkpoint=ckpt)
        trainer.save_model()
        trainer.log_metrics("train", result.metrics)
        trainer.save_metrics("train", result.metrics)
        trainer.save_state()

    # ---- Eval ----
    if training_args.do_eval and val_dataset is not None:
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # ---- Model card ----
    card_kwargs = {
        "finetuned_from": model_args.model_name_or_path,
        "tasks": "nepa-em-pretraining",
        "dataset": data_args.image_dir,
        "tags": ["nepa", "electron-microscopy", "self-supervised", "vision"],
    }
    if training_args.push_to_hub:
        trainer.push_to_hub(**card_kwargs)
    else:
        trainer.create_model_card(**card_kwargs)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
