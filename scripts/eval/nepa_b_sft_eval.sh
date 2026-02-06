# ========================
NGPU=$(python -c "import torch; print(torch.cuda.device_count())")

MODEL_NAME="SixAILab/nepa-base-patch14-224-sft"
DATASET_PATH="data/imagenet-1k-hf"

DATALOADER_NUM_WORKERS=$((4 * NGPU))

# ========================
torchrun \
    --nproc_per_node $NGPU run_image_classification.py \
    \
    --ddp_backend nccl \
    --ddp_find_unused_parameters False \
    \
    --model_name_or_path $MODEL_NAME \
    --dataset_name $DATASET_PATH \
    --load_from_disk True \
    --do_eval True \
    --per_device_eval_batch_size 128 \
    --remove_unused_columns False \
    \
    --bf16 True \
    --dataloader_num_workers $DATALOADER_NUM_WORKERS \
    --dataloader_persistent_workers True \
    --dataloader_pin_memory False \