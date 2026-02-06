#!/bin/bash
# ============================================================
# NEPA-EM  Stage 1 — Tiling pretraining (224×224)
# ============================================================
#
# Step 0:  Pre-tile micrographs (run once, uses 128 CPUs)
# Step 1:  Pretrain NEPA on pre-tiled data (fast I/O)
#
# Hardware: 8×H100 80GB, 192 CPU, 2TB RAM
# ============================================================

set -euo pipefail

# ---------- Paths ----------
DATA_DIR="/data/em_nepa/train"                # 8,823 training micrographs
VAL_DIR="/data/em_nepa/val"                   # 976 validation micrographs
TRAIN_TILES="/data/em_nepa/train_tiles_224"   # pre-tiled output
VAL_TILES="/data/em_nepa/val_tiles_224"       # pre-tiled output
CONFIG="configs/pretrain/nepa-base-patch14-em"
OUTPUT_DIR="/data/ckpt/em_stage1"

# ---------- Hardware ----------
NUM_GPUS=8                           # 8×H100
WORKERS=16                           # dataloader workers per GPU process
PREFETCH=4                           # batches to prefetch per worker

# ---------- Tile ----------
TILE_SIZE=224
TILE_OVERLAP=0.0
TILE_WORKERS=128                     # CPUs for pre-tiling

# ---------- Training ----------
EPOCHS=300
BATCH_PER_GPU=128                    # H100 80GB: 128 tiles @ 224 easily fits
ACCUM=1                              # effective batch = 128 × 8 = 1024
LR=1.5e-4
EMBED_LR=5e-5
WEIGHT_DECAY=0.05
WARMUP_RATIO=0.1
EMA_DECAY=0.9999

# ============================================================
# Step 0: Pre-tile (skip if already done)
# ============================================================

if [ ! -d "${TRAIN_TILES}" ] || [ -z "$(ls -A ${TRAIN_TILES} 2>/dev/null)" ]; then
    echo "============================================================"
    echo "  Pre-tiling training images → ${TRAIN_TILES}"
    echo "  Using ${TILE_WORKERS} CPU workers"
    echo "============================================================"
    python pretile_em_data.py \
        --source_dir "${DATA_DIR}" \
        --output_dir "${TRAIN_TILES}" \
        --tile_size ${TILE_SIZE} \
        --overlap ${TILE_OVERLAP} \
        --num_workers ${TILE_WORKERS} \
        --quality 95
    echo ""
else
    echo "Train tiles already exist: ${TRAIN_TILES} ($(ls ${TRAIN_TILES} | wc -l) files)"
fi

if [ ! -d "${VAL_TILES}" ] || [ -z "$(ls -A ${VAL_TILES} 2>/dev/null)" ]; then
    echo "============================================================"
    echo "  Pre-tiling validation images → ${VAL_TILES}"
    echo "============================================================"
    python pretile_em_data.py \
        --source_dir "${VAL_DIR}" \
        --output_dir "${VAL_TILES}" \
        --tile_size ${TILE_SIZE} \
        --overlap ${TILE_OVERLAP} \
        --num_workers ${TILE_WORKERS} \
        --quality 95
    echo ""
else
    echo "Val tiles already exist: ${VAL_TILES} ($(ls ${VAL_TILES} | wc -l) files)"
fi

# ============================================================
# Step 1: Pretrain NEPA
# ============================================================

echo ""
echo "============================================================"
echo "  NEPA-EM  Stage 1  —  ${TILE_SIZE}×${TILE_SIZE}  pretiled"
echo "  GPUs:    ${NUM_GPUS}×H100"
echo "  Batch:   ${BATCH_PER_GPU} × ${NUM_GPUS} × ${ACCUM} = $(( BATCH_PER_GPU * NUM_GPUS * ACCUM ))"
echo "  Workers: ${WORKERS}/GPU  (prefetch=${PREFETCH})"
echo "  Epochs:  ${EPOCHS}"
echo "  LR:      ${LR}  (embed: ${EMBED_LR})"
echo "============================================================"

torchrun --nproc_per_node=${NUM_GPUS} run_nepa_em.py \
    --image_dir "${TRAIN_TILES}" \
    --val_image_dir "${VAL_TILES}" \
    --pretiled \
    --tile_size ${TILE_SIZE} \
    --config_name ${CONFIG} \
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
    --do_eval \
    --eval_strategy steps \
    --eval_steps 200 \
    --bf16 \
    --tf32 true \
    --dataloader_num_workers ${WORKERS} \
    --dataloader_prefetch_factor ${PREFETCH} \
    --dataloader_persistent_workers true \
    --save_strategy steps \
    --save_steps 200 \
    --save_total_limit 3 \
    --logging_steps 50 \
    --report_to wandb \
    --pos_embed_rescale 3.0 \
    --visualize_embeddings
