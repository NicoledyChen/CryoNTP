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
    train_dataset = EMTilingDataset(
        image_dir=data_args.image_dir,
        tile_size=data_args.tile_size,
        tile_overlap=data_args.tile_overlap,
        random_crop=data_args.random_crop,
        num_random_crops=data_args.num_random_crops,
        transform=train_tf,
        to_rgb=(config.num_channels == 3),
    )

    val_dataset = None
    if data_args.val_image_dir is not None:
        val_dataset = EMTilingDataset(
            image_dir=data_args.val_image_dir,
            tile_size=data_args.tile_size,
            tile_overlap=0.0,   # no overlap for deterministic val
            random_crop=False,
            transform=val_tf,
            to_rgb=(config.num_channels == 3),
        )

    logger.info(f"Train samples: {len(train_dataset)}")
    if val_dataset is not None:
        logger.info(f"Val   samples: {len(val_dataset)}")

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
