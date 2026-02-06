# ========================
export NCCL_DEBUG=INFO
export NCCL_IB_TC=106
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth0
export NCCL_CROSS_NIC=0
export TORCH_DISTRIBUTED_TIMEOUT=1800

: "${WORLD_SIZE:=1}"
: "${RANK:=0}"
: "${MASTER_ADDR:=127.0.0.1}"
: "${MASTER_PORT:=29500}"

# ========================
NGPU=$(python -c "import torch; print(torch.cuda.device_count())")

EXPERIMENT_NAME="nepa-large-patch14-224"
WANDB_PROJECT="Nepa-Pretrain"

CONFIG_NAME="configs/pretrain/nepa-large-patch14-224"
DATASET_PATH="data/imagenet-1k-hf"
OUTPUT_DIR="outputs/${EXPERIMENT_NAME}"

TOTAL_BATCH_SIZE=4096
PER_DEVICE_BATCH_SIZE=128
GRAD_ACCUM_STEPS=$(( TOTAL_BATCH_SIZE / (PER_DEVICE_BATCH_SIZE * NGPU * WORLD_SIZE) ))
NUM_EPOCHS=1600

BASE_LEARNING_RATE=3e-4
LEARNING_RATE=$(python -c "print(${BASE_LEARNING_RATE} * ${TOTAL_BATCH_SIZE} / 256)")

DATALOADER_NUM_WORKERS=$((4 * NGPU))

# ========================
export WANDB_PROJECT=$WANDB_PROJECT

# ========================
torchrun \
    --nnodes=$WORLD_SIZE  \
    --node_rank=$RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    --nproc_per_node $NGPU run_nepa.py \
    \
    --ddp_backend nccl \
    --ddp_find_unused_parameters False \
    \
    --config_name $CONFIG_NAME \
    --image_processor_name  $CONFIG_NAME \
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
    --warmup_ratio 0.025 \
    --weight_decay 0.05 \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --optim adamw_torch \
    \
    --logging_strategy steps \
    --logging_steps 100 \
    --save_strategy steps \
    --save_steps 50000 \
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
