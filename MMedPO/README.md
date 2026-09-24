# [MMedPO: Aligning Medical Vision-Language Models with Clinical-Aware Multimodal Preference Optimization](https://arxiv.org/abs/2412.06141)

[English version see README_EN.md](README_EN.md)

## 💡 概述

<div align=left>
<img src=assets/logo.png width=90% />
</div>

## 📦 环境要求

1. 克隆仓库并进入 MMedPO 目录

```bash
git clone https://github.com/aiming-lab/MMedPO.git
cd MMedPO
```

2. 安装依赖：创建 conda 环境

```Shell
conda create -n MMedPO python=3.10 -y
conda activate MMedPO
pip install --upgrade pip  # enable PEP 660 support
pip install -r requirements.txt
pip install trl
```

3. 从 HuggingFace 下载所需模型权重 [LLaVA-Med-1.5](https://huggingface.co/microsoft/llava-med-v1.5-mistral-7b)。

4. 我们已在 HuggingFace 上发布四个 MMedPO [checkpoint](https://huggingface.co/zky11235/mmedpo_checkpoints)。

5. 医学数据集需先申请使用权限，再下载数据：

- [MIMIC-CXR](https://physionet.org/content/mimic-cxr-jpg/2.0.0/)
- [IU-Xray](https://drive.google.com/file/d/1c0BXEuDy8Cmm2jfN0YYGkQxFZd2ZIoLg/view)（感谢 [R2GenGPT](https://github.com/wang-zhanyu/R2GenGPT) 分享该文件）
- [VQA-RAD](https://osf.io/89kps/)
- [SLAKE](https://www.med-vqa.com/slake/)

## 🪧 数据策展

我们使用 MedKLIP 生成视觉偏好数据，可使用以下命令，或直接运行 `./scripts` 下的 `inference_attention-map_score.sh` 脚本：

```Shell
python ./inference_attention-map_score.py \
    --config ./MedKLIP_config.yaml \
    --model_path /path/to/MedKLIP_model.pth \
    --dataset_name /dataset/name \
    --dataset_type caption \
    --image_root /path/to/dataset/image_folder \
    --annotation_save_root /path/to/save/annotation \
    --noised_image_save_root /path/to/save/noised_image \
```

## 🏋️ 训练

使用 `./scripts` 下的 `train_dpo_visual-text.sh` 脚本，或以下命令；注意指定必要的数据路径与 checkpoint 保存位置：

```
deepspeed --include localhost:0,1,2,3 ./train/dpo/train_dpo_visual-text.py \
    --model_name_or_path /path/to/llava-med_model_checkpoint \
    --deepspeed ./scripts/zero3.json \
    --version v1 \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --data_path /path/to/data_json \
    --image_folder /path/to/img_folder \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir /path/to/output_checkpoint_saving_location \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1\
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 1 \
    --learning_rate 1e-7 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --report_to wandb \
    --tf32 True \
    --model_max_length 1024 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
```

## 🚀 推理

推理脚本位于 `scripts` 目录，指定相关路径后即可运行：

```
bash scripts/inference_llava-med_{vqa/report}.sh
```

## 📚 引用

```bibtex
@article{zhu2024mmedpo,
  title={MMedPO: Aligning Medical Vision-Language Models with Clinical-Aware Multimodal Preference Optimization},
  author={Zhu, Kangyu and Xia, Peng and Li, Yun and Zhu, Hongtu and Wang, Sheng and Yao, Huaxiu},
  journal={arXiv preprint arXiv:2412.06141},
  year={2024}
}
```

## 🙏 致谢

本项目使用了 [LLaVA-Med](https://github.com/microsoft/LLaVA-Med)、[RULE](https://github.com/richard-peng-xia/RULE)、[MedKLIP](https://github.com/MediaBrain-SJTU/MedKLIP) 的代码，感谢作者开源。

---

# 本工作区扩展内容（Scripts 说明）

> 以下内容为在本仓库中新增 / 扩展的脚本说明，补充官方 README 未覆盖的部分。

## 目录导览

| 目录 | 内容 |
|------|------|
| `scripts/` | 训练、推理、评测、数据增强的一键运行脚本与工具 |
| `train/dpo/` | DPO 训练代码（`train_dpo_weighted.py`、`train_dpo_medgemma.py`、`train_dpo_dual_gpu.py`、`llava_trainer_weighted.py`、`dpo_trainer_weighted.py`） |
| `train/rl/` | 强化学习训练（`train_grpo_stage3.py`，GRPO 阶段） |
| `inference/` | 偏好对构建与推理（`build_dpo_pairs_*.py`、`analyze_*`、`generate_master_question_set.py`） |
| `eval/` | 评测脚本（`eval_vqa.py`、`eval_report.py`、`run_eval.sh`、`model_download.py`） |
| `utils/` | 通用工具：tie 权重计算、数据格式转换、DPO 权重重算、LoRA 合并等 |
| `tools/` | 影像处理工具（`remove_lesions_text_clipseg.py`，基于 CLIPSeg 去除病灶区域） |
| `curation/` | 数据策展相关脚本 |
| `rebuttal/` | rebuttal 阶段的补充实验与结果 |
| `assets/` | 图片等资源 |
| `data/` | 已构建的偏好数据集与数据转换脚本 |

## 常用流程

```bash
# 1. 构建偏好对（方法 1：tie 权重）
bash scripts/run_inference_visual_indirect.sh

# 2. 训练
bash scripts/train_sft.sh            # SFT
bash scripts/train_dpo_visual-text.sh  # DPO
bash scripts/train_sspo.sh            # SSPO
bash scripts/train_tie_sspo.sh        # SSPO（动态 w）

# 3. 推理
bash scripts/inference_llava-med_vqa.sh      # VQA
bash scripts/inference_llava-med_report.sh   # 报告生成

# 4. 评测
bash eval/run_eval.sh                        # MedEvalKit 评测
bash scripts/run_compute_iou_slake.sh        # SLAKE IoU
bash scripts/run_evaluate_llmjudge.sh        # LLM-as-judge

# 5. 结果整理
python scripts/plot_slake_results.py
python scripts/evaluate_and_make_latex.py    # 生成 LaTeX 结果表
```

## 调试与诊断脚本

`scripts/` 下另有一批排障脚本，用于在训练异常时定位问题：`check_lora_weights.py`、`debug_model_type.py`、`debug_training_environment.py`、`test_device_check.py`、`test_dpo_device_debug.py`、`test_dual_gpu_dpo.py`、`test_fixed_dpo_trainer.py`、`fix_disable_adapter_issue.py`、`verify_policy_reference_diff.py` 等。

## 数据增强

- `generate_background_randomized.py` — 生成随机化背景（用于构造视觉偏置样本）
- `generate_lesion_subset.py` — 生成病灶子集
- `generate_simulated_iou.py` — 生成模拟 IoU 数据
