#!/bin/bash
export HF_ENDPOINT=https://hf-mirror.com

# Available datasets: MMMU-Medical-test,MMMU-Medical-val,PMC_VQA,MedQA_USMLE,MedMCQA,PubMedQA,OmniMedVQA,Medbullets_op4,Medbullets_op5,MedXpertQA-Text,MedXpertQA-MM,SuperGPQA,HealthBench,IU_XRAY,CheXpert_Plus,MIMIC_CXR,CMB,CMExam,CMMLU,MedQA_MCMLE,VQA_RAD,SLAKE,PATH_VQA,MedFrameQA,Radrestruct
EVAL_DATASETS="SLAKE" 
# Fix: For the VQA_RAD dataset, use the specific JSONL file path instead of the directory path
DATASETS_PATH="/workspace/MMedPO/datasets"
# The DATASETS_PATH should be the parents directory of the dataset directory
# E.g., if the dataset directory is /workspace/MMedPO/datasets/SLAKE, then the DATASETS_PATH should be /workspace/MMedPO/datasets
# And you should rename your slake1.0 folder to SLAKE, this was decided by the EVAL_DATASETS="SLAKE"

OUTPUT_PATH="/workspace/MMedPO/MedEvalKit/Eval_Results/SFT_SLAKE_UCD"
# Available models: TestModel,Qwen2-VL,Qwen2.5-VL,BiMediX2,LLava_Med,Huatuo,InternVL,Llama-3.2,LLava,Janus,HealthGPT,BiomedGPT,Vllm_Text,MedGemma,Med_Flamingo,MedDr
MODEL_NAME="LLava_Med"
# Path to LoRA checkpoint (contains mm_projector.bin or non_lora_trainables.bin)
MODEL_PATH="/workspace/MMedPO/checkpoints/sft_model_lora_SLAKE"

# Now you could use the MODEL_PATH to evaluate the checkpoint
# Path to base model (e.g., llava-med-v1.5-mistral-7b merged or original base)
BASE_MODEL_PATH="/workspace/llava-med-v1.5-mistral-7b"

# BASE_MODEL_PATH could be online path in huggingface too.

# VLLM setting
CUDA_VISIBLE_DEVICES="0,1"
TENSOR_PARALLEL_SIZE="1"
USE_VLLM="False"

# Evaluation setting
SEED=42
REASONING="False"
TEST_TIMES=1

# Model LLM setting
MAX_NEW_TOKENS=8192
MAX_IMAGE_NUM=6
TEMPERATURE=0
TOP_P=0.0001
REPETITION_PENALTY=1

# LLM judge setting - 优化后的配置
USE_LLM_JUDGE="True"
JUDGE_MODEL_TYPE="deepseek"  # openai or gemini or deepseek or claude

# DeepSeek API 配置 - 需要确保账户余额充足
# # gpt api model name
# GPT_MODEL="gpt-4.1-2025-04-14"
GPT_MODEL="deepseek-chat"  # DeepSeek聊天模型
API_KEY="sk-2add7769c362491d851acc6722119b2a"  # 注意：确保账户有足够余额
BASE_URL="https://api.deepseek.com"

# JUDGE_MODEL_TYPE="gemini"  # openai or gemini or deepseek or claude
# GPT_MODEL="gemini-2.5-flash"  # DeepSeek聊天模型
# API_KEY="AIzaSyAqkZ2cavUcwj01mcE_PmsMir3j8s5lWhk"  # 注意：确保账户有足够余额
# BASE_URL="https://generativelanguage.googleapis.com/v1beta/openai/"


# pass hyperparameters and run python sccript
python eval.py \
    --eval_datasets "$EVAL_DATASETS" \
    --datasets_path "$DATASETS_PATH" \
    --output_path "$OUTPUT_PATH" \
    --model_name "$MODEL_NAME" \
    --model_path "$MODEL_PATH" \
    --model_base "$BASE_MODEL_PATH" \
    --seed $SEED \
    --cuda_visible_devices "$CUDA_VISIBLE_DEVICES" \
    --tensor_parallel_size "$TENSOR_PARALLEL_SIZE" \
    --use_vllm "$USE_VLLM" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --max_image_num "$MAX_IMAGE_NUM" \
    --temperature "$TEMPERATURE"  \
    --top_p "$TOP_P" \
    --repetition_penalty "$REPETITION_PENALTY" \
    --reasoning "$REASONING" \
    --use_llm_judge "$USE_LLM_JUDGE" \
    --judge_model_type "$JUDGE_MODEL_TYPE" \
    --judge_model "$GPT_MODEL" \
    --api_key "$API_KEY" \
    --base_url "$BASE_URL" \
    --test_times "$TEST_TIMES" \
