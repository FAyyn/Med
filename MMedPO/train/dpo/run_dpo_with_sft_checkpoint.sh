#!/bin/bash

# DPO Training Script using SFT Checkpoint
# This script performs Direct Preference Optimization using an existing SFT checkpoint
# and the specified base model, image folder, and DPO data path

set -e  # Exit on any error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')] $1${NC}"
}

print_success() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] $1${NC}"
}

print_error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] $1${NC}"
}

# Configuration - Using the paths provided by user
BASE_MODEL_PATH="/workspace/llava-med-v1.5-mistral-7b"
SFT_CHECKPOINT_PATH="/workspace/MMedPO/MMedPO/checkpoints/sft_model_lora"
IMAGE_FOLDER="/workspace/MMedPO/datasets/Slake1.0/imgs"
DPO_DATA_PATH="/workspace/MMedPO/MMedPO/data/tie_results_with_dpo_weight.json"
OUTPUT_DIR="/workspace/MMedPO/MMedPO/checkpoints/dpo_model_from_sft"

print_status "Starting DPO Training with SFT Checkpoint"
print_status "=========================================="
print_status "Base Model: $BASE_MODEL_PATH"
print_status "SFT Checkpoint: $SFT_CHECKPOINT_PATH"
print_status "Image Folder: $IMAGE_FOLDER"
print_status "DPO Data: $DPO_DATA_PATH"
print_status "Output Directory: $OUTPUT_DIR"

# Verify paths exist
if [[ ! -d "$BASE_MODEL_PATH" ]]; then
    print_error "Base model path does not exist: $BASE_MODEL_PATH"
    exit 1
fi

if [[ ! -d "$SFT_CHECKPOINT_PATH" ]]; then
    print_error "SFT checkpoint path does not exist: $SFT_CHECKPOINT_PATH"
    exit 1
fi

if [[ ! -d "$IMAGE_FOLDER" ]]; then
    print_error "Image folder does not exist: $IMAGE_FOLDER"
    exit 1
fi

if [[ ! -f "$DPO_DATA_PATH" ]]; then
    print_error "DPO data file does not exist: $DPO_DATA_PATH"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"
print_success "Created output directory: $OUTPUT_DIR"

# Set environment variables
export CUDA_VISIBLE_DEVICES=1,2,3
export PYTHONPATH="/workspace/MMedPO/MMedPO/train/dpo:$PYTHONPATH"
# Disable wandb completely
export WANDB_MODE=disabled
export WANDB_DISABLED=true
export WANDB_SILENT=true

# Change to the training directory
cd /workspace/MMedPO/MMedPO/train/dpo

print_status "Starting DPO training with torchrun..."

# Run DPO training using the SFT checkpoint
WANDB_MODE=disabled WANDB_DISABLED=true WANDB_SILENT=true torchrun --nproc_per_node=3 --master_port=$((RANDOM + 30000)) train_dpo_weighted.py \
    --model_name_or_path "$BASE_MODEL_PATH" \
    --lora_checkpoint_path "$SFT_CHECKPOINT_PATH" \
    --data_path "$DPO_DATA_PATH" \
    --image_folder "$IMAGE_FOLDER" \
    --output_dir "$OUTPUT_DIR" \
    --version v1 \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50 \
    --max_steps 500 \
    --save_total_limit 3 \
    --learning_rate 2e-6 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 0 \
    --lazy_preprocess True \
    --report_to none \
    --ddp_find_unused_parameters False \
    --max_grad_norm 1.0 \
    --remove_unused_columns False \
    --lora_enable True \
    --lora_r 128 \
    --lora_alpha 256 \
    --lora_dropout 0.05 \
    --lora_weight_path "" \
    --lora_bias "none" \
    --mm_projector_lr 2e-5 \
    --loss_use_weight True \
    --bits 4 \
    --double_quant True \
    --quant_type nf4

# Check if training completed successfully
if [[ $? -eq 0 ]]; then
    print_success "DPO training completed successfully!"
    print_success "Model saved to: $OUTPUT_DIR"
    print_status "Training Configuration Summary:"
    print_status "  - Base Model: $BASE_MODEL_PATH"
    print_status "  - SFT Checkpoint: $SFT_CHECKPOINT_PATH"
    print_status "  - LoRA Parameters: r=128, alpha=256, dropout=0.05"
    print_status "  - Quantization: 4-bit with NF4"
    print_status "  - DPO Beta: 0.5 (enhanced), Learning Rate: 2e-6, Max Steps: 500"
else
    print_error "DPO training failed!"
    exit 1
fi

print_success "DPO training pipeline completed at $(date)"