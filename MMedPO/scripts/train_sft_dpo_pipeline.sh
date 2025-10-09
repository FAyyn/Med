#!/bin/bash

# Two-Stage Training Pipeline: SFT + DPO
# This script performs supervised fine-tuning followed by direct preference optimization

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

# Configuration
CONDA_ENV_NAME="MMedPO"
CONDA_PATH="/workspace/miniconda3"
SCRIPT_DIR="/workspace/MMedPO/MMedPO/scripts"
PROJECT_ROOT="/workspace/MMedPO/MMedPO"

# Training parameters
SFT_DATA_PATH="/workspace/MMedPO/MMedPO/data/slake_dpo_weighted.json"
DPO_DATA_PATH="/workspace/MMedPO/MMedPO/data/tie_results_with_dpo_weight.json"
IMAGE_FOLDER="/workspace/MMedPO/datasets/Slake1.0/imgs"
SFT_OUTPUT_DIR="/workspace/MMedPO/MMedPO/checkpoints/sft_model"
DPO_OUTPUT_DIR="/workspace/MMedPO/MMedPO/checkpoints/dpo_model"

print_status "Starting Two-Stage Training Pipeline (SFT + DPO)"
print_status "================================================"

# Set offline mode for Hugging Face Hub to avoid network issues
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Optimize CUDA memory allocation to reduce fragmentation
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Initialize conda
print_status "Initializing conda environment..."
source $CONDA_PATH/etc/profile.d/conda.sh

# Check if conda environment exists
if conda env list | grep -q "^$CONDA_ENV_NAME "; then
    print_success "Conda environment '$CONDA_ENV_NAME' found"
else
    print_error "Conda environment '$CONDA_ENV_NAME' not found!"
    print_error "Available environments:"
    conda env list
    exit 1
fi

# Activate conda environment
print_status "Activating conda environment: $CONDA_ENV_NAME"
conda activate $CONDA_ENV_NAME

# Verify environment activation
if [[ "$CONDA_DEFAULT_ENV" == "$CONDA_ENV_NAME" ]]; then
    print_success "Successfully activated environment: $CONDA_DEFAULT_ENV"
else
    print_error "Failed to activate environment: $CONDA_ENV_NAME"
    exit 1
fi

# Change to project directory
cd $PROJECT_ROOT
print_status "Changed to project directory: $(pwd)"

# Verify data files exist
print_status "Verifying data files..."
if [[ ! -f "$SFT_DATA_PATH" ]]; then
    print_error "SFT dataset not found: $SFT_DATA_PATH"
    exit 1
fi

if [[ ! -f "$DPO_DATA_PATH" ]]; then
    print_error "DPO dataset not found: $DPO_DATA_PATH"
    exit 1
fi

if [[ ! -d "$IMAGE_FOLDER" ]]; then
    print_error "Image folder not found: $IMAGE_FOLDER"
    exit 1
fi

print_success "All data files verified"

# Create checkpoint directories
mkdir -p "$SFT_OUTPUT_DIR"
mkdir -p "$DPO_OUTPUT_DIR"

# Stage 1: Supervised Fine-Tuning (SFT)
print_status "================================================"
print_status "Stage 1: Starting Supervised Fine-Tuning (SFT)"
print_status "================================================"
print_status "SFT Dataset: $SFT_DATA_PATH"
print_status "Output Directory: $SFT_OUTPUT_DIR"

# Set environment variables for SFT
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Run SFT training
deepspeed /workspace/MMedPO/MMedPO/train/dpo/llava/train/train.py \
    --deepspeed /workspace/MMedPO/MMedPO/train/dpo/scripts/zero2.json \
    --model_name_or_path "/workspace/llava-med-v1.5-mistral-7b" \
    --version v1 \
    --data_path "$SFT_DATA_PATH" \
    --image_folder "$IMAGE_FOLDER" \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir "$SFT_OUTPUT_DIR" \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 3 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 64 \
    --gradient_checkpointing True \
    --dataloader_num_workers 0 \
    --lazy_preprocess True \
    --report_to none \
    --ddp_find_unused_parameters False \
    --max_grad_norm 1.0

# Check if SFT training completed successfully
if [[ $? -eq 0 ]]; then
    print_success "SFT training completed successfully!"
    print_success "SFT model saved to: $SFT_OUTPUT_DIR"
else
    print_error "SFT training failed!"
    exit 1
fi

# Wait a moment before starting DPO
sleep 5

# Stage 2: Direct Preference Optimization (DPO)
print_status "================================================"
print_status "Stage 2: Starting Direct Preference Optimization (DPO)"
print_status "================================================"
print_status "DPO Dataset: $DPO_DATA_PATH"
print_status "SFT Model: $SFT_OUTPUT_DIR"
print_status "Output Directory: $DPO_OUTPUT_DIR"

# Switch to DPO conda environment
print_status "Switching to DPO conda environment: MMedPO_check_point"
conda deactivate
conda activate MMedPO_check_point

# Verify DPO environment activation
if [[ "$CONDA_DEFAULT_ENV" == "MMedPO_check_point" ]]; then
    print_success "Successfully activated DPO environment: $CONDA_DEFAULT_ENV"
else
    print_error "Failed to activate DPO environment: MMedPO_check_point"
    exit 1
fi

# Set environment variables for DPO
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Run DPO training
print_status "Starting DPO training..."
deepspeed /workspace/MMedPO/MMedPO/train/dpo/train_dpo_visual-text.py \
    --deepspeed /workspace/MMedPO/MMedPO/train/dpo/scripts/zero2.json \
    --model_name_or_path "$SFT_OUTPUT_DIR" \
    --data_path "$DPO_DATA_PATH" \
    --image_folder "$IMAGE_FOLDER" \
    --output_dir "$DPO_OUTPUT_DIR" \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 3 \
    --learning_rate 5e-7 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 64 \
    --gradient_checkpointing True \
    --dataloader_num_workers 0 \
    --lazy_preprocess True \
    --report_to none \
    --ddp_find_unused_parameters False \
    --max_grad_norm 1.0 \
    --beta 0.1 \
    --dpo_alpha 1.0 \
    --remove_unused_columns False \
    --lora_enable True \
    --lora_r 128 \
    --lora_alpha 256 \
    --lora_dropout 0.05 \
    --lora_weight_path "" \
    --lora_bias "none" \
    --mm_projector_lr 2e-5

# Check if DPO training completed successfully
if [[ $? -eq 0 ]]; then
    print_success "DPO training completed successfully!"
    print_success "DPO model saved to: $DPO_OUTPUT_DIR"
else
    print_error "DPO training failed!"
    exit 1
fi

# Training pipeline completed
print_status "================================================"
print_success "Two-Stage Training Pipeline Completed Successfully!"
print_status "================================================"
print_success "SFT Model: $SFT_OUTPUT_DIR"
print_success "Final DPO Model: $DPO_OUTPUT_DIR"
print_status "Training completed without external logging services"

# Deactivate conda environment
conda deactivate
print_status "Conda environment deactivated"

print_success "Training pipeline finished at $(date)"