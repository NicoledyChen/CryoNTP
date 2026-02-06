#!/bin/bash
# ============================================================
# NEPA-EM  Stage 2 — Progressive resolution pretraining
# ============================================================
#
# For each resolution:
#   1.  Pre-tile micrographs at that resolution (once)
#   2.  Continue pretraining from previous checkpoint
#
# Resolution → sequence length:
#   224  → 16×16 =   256 patches   (done in Stage 1)
#   336  → 24×24 =   576 patches
#   448  → 32×32 = 1 024 patches
#   672  → 48×48 = 2 304 patches
#
# Hardware: 8×H100 80GB, 192 CPU, 2TB RAM
# ============================================================

set -euo pipefail

# ---------- Paths ----------
DATA_DIR="/data/em_nepa/train"               # source micrographs
VAL_DIR="/data/em_nepa/val"
CONFIG="configs/pretrain/nepa-base-patch14-em"
STAGE1_CKPT="/data/ckpt/em_stage1/checkpoint-7400"
BASE_OUTPUT="/data/ckpt/em_stage2"
TILE_BASE="/data/em_nepa"                    # pre-tiled data root

# ---------- Hardware ----------
NUM_GPUS=8
TILE_WORKERS=128                             # CPUs for pre-tiling

# ---------- Resolution schedule ----------
# Format:  "resolution:batch:workers:epochs:lr"
# Batch & workers tuned per resolution for H100 80GB memory.
PHASES=(
    "336:64:16:100:1e-4"
    "448:32:12:100:5e-5"
    "672:8:8:50:2e-5"
)

ACCUM=4
WEIGHT_DECAY=0.05
WARMUP_RATIO=0.1
EMA_DECAY=0.99999
PREFETCH=4

# ============================================================

PREV_CKPT="${STAGE1_CKPT}"

for PHASE in "${PHASES[@]}"; do
    IFS=":" read -r RES BATCH WORKERS EPOCHS LR <<< "${PHASE}"

    TRAIN_TILES="${TILE_BASE}/train_tiles_${RES}"
    VAL_TILES="${TILE_BASE}/val_tiles_${RES}"
    OUTPUT_DIR="${BASE_OUTPUT}_${RES}"

    echo ""
    echo "============================================================"
    echo "  Stage 2  —  ${RES}×${RES}"
    echo "  Batch:   ${BATCH} × ${NUM_GPUS} × ${ACCUM} = $(( BATCH * NUM_GPUS * ACCUM ))"
    echo "  Workers: ${WORKERS}/GPU"
    echo "  From:    ${PREV_CKPT}"
    echo "  Output:  ${OUTPUT_DIR}"
    echo "============================================================"

    # ---- Pre-tile at this resolution ----
    if [ ! -d "${TRAIN_TILES}" ] || [ -z "$(ls -A ${TRAIN_TILES} 2>/dev/null)" ]; then
        echo "  Pre-tiling train → ${TRAIN_TILES} ..."
        python pretile_em_data.py \
            --source_dir "${DATA_DIR}" \
            --output_dir "${TRAIN_TILES}" \
            --tile_size ${RES} --overlap 0.0 \
            --num_workers ${TILE_WORKERS} --quality 95
    else
        echo "  Train tiles exist: $(ls ${TRAIN_TILES} | wc -l) files"
    fi

    if [ ! -d "${VAL_TILES}" ] || [ -z "$(ls -A ${VAL_TILES} 2>/dev/null)" ]; then
        echo "  Pre-tiling val → ${VAL_TILES} ..."
        python pretile_em_data.py \
            --source_dir "${VAL_DIR}" \
            --output_dir "${VAL_TILES}" \
            --tile_size ${RES} --overlap 0.0 \
            --num_workers ${TILE_WORKERS} --quality 95
    else
        echo "  Val tiles exist: $(ls ${VAL_TILES} | wc -l) files"
    fi

    # ---- Train ----
    torchrun --nproc_per_node=${NUM_GPUS} run_nepa_em.py \
        --image_dir "${TRAIN_TILES}" \
        --val_image_dir "${VAL_TILES}" \
        --pretiled \
        --tile_size ${RES} \
        --config_name ${CONFIG} \
        --model_name_or_path ${PREV_CKPT} \
        --output_dir ${OUTPUT_DIR} \
        --per_device_train_batch_size ${BATCH} \
        --gradient_accumulation_steps ${ACCUM} \
        --learning_rate ${LR} \
        --weight_decay ${WEIGHT_DECAY} \
        --warmup_ratio ${WARMUP_RATIO} \
        --ema_decay ${EMA_DECAY} \
        --num_train_epochs ${EPOCHS} \
        --do_train \
        --do_eval \
        --eval_strategy steps \
        --eval_steps 1000 \
        --bf16 \
        --tf32 true \
        --dataloader_num_workers ${WORKERS} \
        --dataloader_prefetch_factor ${PREFETCH} \
        --dataloader_persistent_workers true \
        --save_strategy steps \
        --save_steps 1000 \
        --save_total_limit 2 \
        --logging_steps 50 \
        --report_to wandb \
        --overwrite_output_dir \
        --pos_embed_rescale 3.0

    PREV_CKPT="${OUTPUT_DIR}"
done

echo ""
echo "============================================================"
echo "  Progressive resolution pretraining complete!"
echo "  Final checkpoint: ${PREV_CKPT}"
echo "============================================================"
