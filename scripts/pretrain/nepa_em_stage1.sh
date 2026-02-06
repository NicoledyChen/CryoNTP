#!/bin/bash
# ============================================================
# NEPA-EM  Stage 1 — Tiling pretraining (224×224)
# ============================================================
#
# Pre-train NEPA on small tiles extracted from large EM images.
# This is the first stage: the model learns local spatial
# relationships and texture representations at native resolution.
#
# Adjust DATA_DIR / VAL_DIR / NUM_GPUS to match your setup.
# ============================================================

set -euo pipefail

# ---------- Paths ----------
DATA_DIR="/data/em_nepa/train"       # 8,823 training micrographs
VAL_DIR="/data/em_nepa/val"          # 976 validation micrographs
CONFIG="configs/pretrain/nepa-base-patch14-em"
OUTPUT_DIR="/data/ckpt/em_stage1"

# ---------- Hardware ----------
NUM_GPUS=8                           # number of GPUs

# ---------- Tile ----------
TILE_SIZE=224                        # tile edge in pixels  (224/14 = 16×16 = 256 patches)
TILE_OVERLAP=0.0                     # 0.0 = no overlap;  0.5 = 50 % overlap

# ---------- Training ----------
EPOCHS=300
BATCH_PER_GPU=64                     # per-device batch size
ACCUM=1                              # gradient accumulation steps
LR=1.5e-4                            # base learning rate
EMBED_LR=5e-5                        # separate LR for patch-embedding layer
WEIGHT_DECAY=0.05
WARMUP_RATIO=0.1
EMA_DECAY=0.9999

# ---------- Build command ----------
CMD="torchrun --nproc_per_node=${NUM_GPUS} run_nepa_em.py \
    --image_dir ${DATA_DIR} \
    --config_name ${CONFIG} \
    --tile_size ${TILE_SIZE} \
    --tile_overlap ${TILE_OVERLAP} \
    --output_dir ${OUTPUT_DIR} \
    --per_device_train_batch_size ${BATCH_PER_GPU} \
    --gradient_accumulation_steps ${ACCUM} \
    --learning_rate ${LR} \
    --embed_lr ${EMBED_LR} \
    --weight_decay ${WEIGHT_DECAY} \
    --warmup_ratio ${WARMUP_RATIO} \
    --ema_decay ${EMA_DECAY} \
    --num_train_epochs ${EPOCHS} \
    --do_train \
    --bf16 \
    --dataloader_num_workers 8 \
    --save_strategy epoch \
    --save_total_limit 3 \
    --logging_steps 50 \
    --report_to wandb \
    --pos_embed_rescale 3.0"

# Optional: add validation
if [ -n "${VAL_DIR}" ] && [ -d "${VAL_DIR}" ]; then
    CMD="${CMD} --val_image_dir ${VAL_DIR} --do_eval --eval_strategy epoch"
fi

echo "============================================================"
echo "NEPA-EM  Stage 1  —  ${TILE_SIZE}×${TILE_SIZE}  tiling"
echo "  Data:       ${DATA_DIR}"
echo "  Output:     ${OUTPUT_DIR}"
echo "  GPUs:       ${NUM_GPUS}"
echo "  Batch:      ${BATCH_PER_GPU} × ${NUM_GPUS} × ${ACCUM}"
echo "  Epochs:     ${EPOCHS}"
echo "  LR:         ${LR}  (embed: ${EMBED_LR})"
echo "============================================================"

eval ${CMD}
