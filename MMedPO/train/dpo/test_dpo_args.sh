#!/bin/bash

# Test script to verify DPO training arguments
# This script will test the argument parsing without actually running training

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')] $1${NC}"
}

print_success() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] $1${NC}"
}

print_error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] $1${NC}"
}

# Configuration
BASE_MODEL_PATH="/workspace/llava-med-v1.5-mistral-7b"
SFT_CHECKPOINT_PATH="/workspace/MMedPO/MMedPO/checkpoints/sft_model_lora"
IMAGE_FOLDER="/workspace/MMedPO/datasets/Slake1.0/imgs"
DPO_DATA_PATH="/workspace/MMedPO/MMedPO/data/tie_results_with_dpo_weight.json"
OUTPUT_DIR="/workspace/MMedPO/MMedPO/checkpoints/dpo_model_from_sft_test"

print_status "Testing DPO Training Arguments"
print_status "==============================="

# Set environment variables to disable wandb
export WANDB_MODE=disabled
export WANDB_DISABLED=true
export WANDB_SILENT=true
export PYTHONPATH="/workspace/MMedPO/MMedPO/train/dpo:$PYTHONPATH"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Change to the training directory
cd /workspace/MMedPO/MMedPO/train/dpo

print_status "Testing argument parsing..."

# Test with --help to see if the script accepts our arguments
python3 train_dpo_weighted.py --help > /tmp/dpo_help.txt 2>&1 || true

print_status "Checking if our arguments are supported..."

# Test with dry run (just parse arguments, don't train)
python3 -c "
import sys
sys.path.append('/workspace/MMedPO/MMedPO/train/dpo')
from train_dpo_weighted import ModelArguments, DataArguments, TrainingArguments
import transformers

# Test argument parsing
parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))

# Test arguments
test_args = [
    '--model_name_or_path', '$BASE_MODEL_PATH',
    '--lora_checkpoint_path', '$SFT_CHECKPOINT_PATH',
    '--data_path', '$DPO_DATA_PATH',
    '--image_folder', '$IMAGE_FOLDER',
    '--output_dir', '$OUTPUT_DIR',
    '--version', 'v1',
    '--vision_tower', 'openai/clip-vit-large-patch14-336',
    '--mm_projector_type', 'mlp2x_gelu',
    '--mm_vision_select_layer', '-2',
    '--mm_use_im_start_end', 'False',
    '--mm_use_im_patch_token', 'False',
    '--image_aspect_ratio', 'pad',
    '--group_by_modality_length', 'True',
    '--bf16', 'True',
    '--num_train_epochs', '1',
    '--per_device_train_batch_size', '1',
    '--gradient_accumulation_steps', '2',
    '--evaluation_strategy', 'no',
    '--save_strategy', 'steps',
    '--save_steps', '500',
    '--learning_rate', '2e-6',
    '--weight_decay', '0.',
    '--warmup_ratio', '0.03',
    '--lr_scheduler_type', 'cosine',
    '--logging_steps', '1',
    '--tf32', 'True',
    '--model_max_length', '2048',
    '--gradient_checkpointing', 'True',
    '--dataloader_num_workers', '0',
    '--lazy_preprocess', 'True',
    '--report_to', 'none',
    '--remove_unused_columns', 'False',
    '--lora_enable', 'True',
    '--lora_r', '128',
    '--lora_alpha', '256',
    '--lora_dropout', '0.05',
    '--lora_weight_path', '',
    '--lora_bias', 'none',
    '--mm_projector_lr', '2e-5',
    '--loss_use_weight', 'True',
    '--bits', '4',
    '--double_quant', 'True',
    '--quant_type', 'nf4'
]

try:
    # Use parse_args_into_dataclasses like the original script
    model_args, data_args, training_args = parser.parse_args_into_dataclasses(test_args)
    print('✅ Argument parsing successful!')
    print(f'Model path: {model_args.model_name_or_path}')
    print(f'LoRA checkpoint: {model_args.lora_checkpoint_path}')
    print(f'Data path: {data_args.data_path}')
    print(f'Image folder: {data_args.image_folder}')
    print(f'Output dir: {training_args.output_dir}')
    print(f'LoRA enabled: {training_args.lora_enable}')
    print(f'LoRA r: {training_args.lora_r}')
except Exception as e:
    print(f'❌ Argument parsing failed: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
"

if [[ $? -eq 0 ]]; then
    print_success "✅ All arguments are supported and parsed correctly!"
    print_status "You can now run the full DPO training script."
else
    print_error "❌ Argument parsing failed. Please check the error messages above."
    exit 1
fi

print_success "Test completed successfully!"