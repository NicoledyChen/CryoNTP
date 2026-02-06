#!/bin/bash
# ============================================================
# NEPA-EM  Stage 2 — Progressive resolution pretraining
# ============================================================
#
# Resume from the Stage-1 checkpoint and progressively increase
# the tile resolution.  Each resolution phase loads the previous
# checkpoint.
#
# Because NEPA uses RoPE (dynamic, coordinate-based positional
# encoding), the model transfers across resolutions with zero
# architecture change — only the sequence length grows:
#
#   224  → 16×16 =   256 patches
#   336  → 24×24 =   576 patches
#   448  → 32×32 = 1 024 patches
#   672  → 48×48 = 2 304 patches
#
# Adjust the RESOLUTIONS array and per-resolution hyperparameters
# to match your GPU memory / schedule.
# ============================================================

set -euo pipefail

# ---------- Paths ----------
DATA_DIR="/data/em_nepa/train"               # 8,823 training micrographs
VAL_DIR="/data/em_nepa/val"                  # 976 validation micrographs
CONFIG="configs/pretrain/nepa-base-patch14-em"
STAGE1_CKPT="/data/ckpt/em_stage1"           # Stage 1 output
BASE_OUTPUT="/data/ckpt/em_stage2"

# ---------- Hardware ----------
NUM_GPUS=4

# ---------- Resolution schedule ----------
# Format:  "resolution:batch:epochs:lr"
# Batch is per-device; effective batch = batch × NUM_GPUS × ACCUM.
# Lower batch & LR as resolution grows to fit in GPU memory.
PHASES=(
    "336:32:100:1e-4"
    "448:16:100:5e-5"
    "672:4:50:2e-5"
)

ACCUM=4
WEIGHT_DECAY=0.05
WARMUP_RATIO=0.1
EMA_DECAY=0.99999     # smoother EMA for fine-tuning stages

# ============================================================

PREV_CKPT="${STAGE1_CKPT}"

for PHASE in "${PHASES[@]}"; do
    IFS=":" read -r RES BATCH EPOCHS LR <<< "${PHASE}"
    OUTPUT_DIR="${BASE_OUTPUT}_${RES}"

    echo ""
    echo "============================================================"
    echo "  Stage 2  —  ${RES}×${RES}  (batch=${BATCH}, lr=${LR}, epochs=${EPOCHS})"
    echo "  Loading from: ${PREV_CKPT}"
    echo "  Output:       ${OUTPUT_DIR}"
    echo "============================================================"

    CMD="torchrun --nproc_per_node=${NUM_GPUS} run_nepa_em.py \
        --image_dir ${DATA_DIR} \
        --config_name ${CONFIG} \
        --model_name_or_path ${PREV_CKPT} \
        --tile_size ${RES} \
        --tile_overlap 0.0 \
        --output_dir ${OUTPUT_DIR} \
        --per_device_train_batch_size ${BATCH} \
        --gradient_accumulation_steps ${ACCUM} \
        --learning_rate ${LR} \
        --weight_decay ${WEIGHT_DECAY} \
        --warmup_ratio ${WARMUP_RATIO} \
        --ema_decay ${EMA_DECAY} \
        --num_train_epochs ${EPOCHS} \
        --do_train \
        --bf16 \
        --dataloader_num_workers 8 \
        --save_strategy epoch \
        --save_total_limit 2 \
        --logging_steps 50 \
        --report_to wandb \
        --overwrite_output_dir \
        --pos_embed_rescale 3.0"

    # Optional: add validation
    if [ -n "${VAL_DIR}" ] && [ -d "${VAL_DIR}" ]; then
        CMD="${CMD} --val_image_dir ${VAL_DIR} --do_eval --eval_strategy epoch"
    fi

    eval ${CMD}

    PREV_CKPT="${OUTPUT_DIR}"
done

echo ""
echo "============================================================"
echo "  Progressive resolution pretraining complete!"
echo "  Final checkpoint: ${PREV_CKPT}"
echo "============================================================"
