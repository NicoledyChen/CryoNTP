# ========================
NGPU=$(python -c "import torch; print(torch.cuda.device_count())")

EXPERIMENT_NAME="nepa-base-patch14-224-sft"
WANDB_PROJECT="Nepa-SFT"

MODEL_NAME="SixAILab/nepa-base-patch14-224-sft"
MODEL_REVISION="init"
DATASET_PATH="data/imagenet-1k-hf"
OUTPUT_DIR="outputs/${EXPERIMENT_NAME}"

TOTAL_BATCH_SIZE=1024
PER_DEVICE_BATCH_SIZE=128
GRAD_ACCUM_STEPS=$((TOTAL_BATCH_SIZE / (PER_DEVICE_BATCH_SIZE * NGPU)))
NUM_EPOCHS=100
BASE_LEARNING_RATE=1.5e-3
LEARNING_RATE=$(python -c "print(${BASE_LEARNING_RATE} * ${TOTAL_BATCH_SIZE} / 256)")

DATALOADER_NUM_WORKERS=$((4 * NGPU))

# ========================
export WANDB_PROJECT=$WANDB_PROJECT

# ========================
torchrun \
    --nproc_per_node $NGPU run_image_classification.py \
    \
    --ddp_backend nccl \
    --ddp_find_unused_parameters False \
    \
    --model_name_or_path $MODEL_NAME \
    --model_revision $MODEL_REVISION \
    --freeze_vit False \
    --freeze_embed True \
    \
    --dataset_name $DATASET_PATH \
    --load_from_disk True \
    --dataloader_drop_last True \
    \
    --do_train \
    --output_dir $OUTPUT_DIR \
    --remove_unused_columns False \
    \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_train_batch_size $PER_DEVICE_BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --learning_rate $LEARNING_RATE \
    --lr_scheduler_type cosine \
    --lr_scheduler_kwargs '{"custom_scheduler_type": "llrd_cosine_warmup"}' \
    --warmup_ratio 0.20 \
    --weight_decay 0.05 \
    --adam_beta1 0.9 \
    --adam_beta2 0.999 \
    --optim adamw_torch \
    --llrd 0.65 \
    --ema_decay 0.9999 \
    \
    --logging_strategy steps \
    --logging_steps 100 \
    --eval_strategy epoch \
    --save_strategy epoch \
    --load_best_model_at_end True \
    --metric_for_best_model eval_ema_accuracy \
    --save_total_limit 1 \
    \
    --seed 1337 \
    --bf16 True \
    \
    --dataloader_num_workers $DATALOADER_NUM_WORKERS \
    --dataloader_persistent_workers True \
    --dataloader_pin_memory False \
    \
    --report_to wandb \
    --run_name $EXPERIMENT_NAME
