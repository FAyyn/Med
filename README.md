# Med — 医疗多模态大模型偏好优化研究仓库

本仓库整合了 **MMedPO**（临床感知的多模态偏好优化）及其后续工作 **CaMedPO / CaMPPO**（因果感知偏好优化）的全部代码、评测工具、实验分析与论文资料。

- 主项目论文：MMedPO — [arXiv:2412.06141](https://arxiv.org/abs/2412.06141)
- 后续方法论文：*Causality-Aware Preference Optimization for Aligning Medical Vision Language Models*
- 评测框架：MedEvalKit — [arXiv:2506.07044](https://arxiv.org/abs/2506.07044)

---

## 仓库结构

| 目录 | 功能 |
|------|------|
| [`MMedPO/`](MMedPO/README.md) | 主项目：数据构建、SFT / DPO / SSPO / GRPO 训练、推理、评测与工具脚本 |
| [`MedEvalKit/`](MedEvalKit/Readme.md) | 医疗多模态大模型统一评测框架 |
| [`baselines/`](baselines/README.md) | 对比方法实现（SimPO 等） |
| [`evaluation/`](evaluation/README.md) | 评测工具：RadGraph、CheXbert、MedGemma |
| [`analysis/`](analysis/README.md) | 实验结果分析、案例分类与可视化脚本 |
| [`literature/`](literature/README.md) | 参考文献 PDF |

---

## 快速开始

### 1. 环境安装

```bash
# 主项目（MMedPO）
conda create -n MMedPO python=3.10 -y
conda activate MMedPO
pip install --upgrade pip
pip install -r MMedPO/requirements.txt
pip install trl

# 评测框架（MedEvalKit）
pip install -r MedEvalKit/requirements.txt
```

模型权重需自行下载：基础模型 [LLaVA-Med-1.5](https://huggingface.co/microsoft/llava-med-v1.5-mistral-7b)，MMedPO 官方 checkpoint 见 [mmedpo_checkpoints](https://huggingface.co/zky11235/mmedpo_checkpoints)。

### 2. 数据准备

- **SLAKE 数据集**与处理好的遮挡背景图：[Google Drive](https://drive.google.com/file/d/1YJC7KspZohlfGxylWAKc6bVxfMZKGbYS/view?usp=drive_link)
- 已构建好的偏好数据在 `MMedPO/data/` 下，可直接使用：
  - `slake_dpo_weighted.json` — SLAKE 加权 DPO 数据
  - `tie_dpo_dataset_method1_converted.json` — 方法 1（tie 权重）生成的 DPO 数据
  - `iuxray_dpo_weighted.json`、`mimic_dpo_weighted.json` — IU-Xray / MIMIC 数据
  - `iuxray_sft_dataset.json` — SFT 格式数据

### 3. 训练

所有训练脚本位于 `MMedPO/scripts/`，参数（数据路径、模型路径、GPU 调用等）可直接在脚本内修改，然后 `bash MMedPO/scripts/xxx.sh` 运行。

| 脚本 | 用途 |
|------|------|
| `train_sft.sh` | SFT 训练，可直接使用转成 SFT 格式的 DPO 数据 |
| `train_dpo.sh` | DPO 训练 |
| `train_dpo_visual-text.sh` / `train_dpo_visual-text_optimized.sh` | 视觉-文本 DPO 训练（后者为优化版） |
| `train_sspo.sh` | 标准 SSPO |
| `train_sspo_adv.sh` | SSPO：用权重替代动态 w 参与 loss 计算 |
| `train_tie_sspo.sh` | SSPO：加入动态 w 计算 |
| `run_sppo.py` | SSPO 训练入口 |
| `MMedPO/train/rl/train_grpo_stage3.py` | GRPO 强化学习阶段训练 |

### 4. 偏好对构建与推理

| 脚本 | 用途 |
|------|------|
| `run_inference_visual_indirect.sh` | 方法 1：以 `slake_dpo_weighted.json` 为输入做 tie 计算并构建偏好对 |
| `run_inference_text_contrast.sh` | 文本对比类推理 |
| `run_inference_visual_consistency.sh` | 视觉一致性类推理 |
| `run_full_inference_background.sh` | 背景随机化全量推理 |
| `inference_llava-med_vqa.sh` / `inference_llava-med_report.sh` | LLaVA-Med 的 VQA / 报告生成推理 |
| `inference_attention-map_score.sh` | 注意力图打分推理 |

### 5. 评测

- **MedEvalKit**：`MedEvalKit/eval.py` + `eval.sh` / `eval_chunked.sh`，详细说明见 [`MedEvalKit/Readme.md`](MedEvalKit/Readme.md)
- **SLAKE 与 LLM-as-judge**：`eval_slake_inference.py`、`eval_slake_iou.py`、`eval_slake_combined.py`、`run_evaluate_llmjudge.sh`
- **IoU 计算**：`run_compute_iou_slake.sh`（配合 `compute_iou_slake.py`）
- **指标工具**：`evaluation/` 下的 RadGraph、CheXbert（见 [`evaluation/README.md`](evaluation/README.md)）

### 6. 结果分析

`analysis/` 下脚本用于对比 baseline / MMedPO / CaMedPO 的表现、按类别整理案例并生成可视化页面，详见 [`analysis/README.md`](analysis/README.md)。

---

## 说明

- 评测脚本已补充 checkpoint 支持，可直接加载 checkpoint 进行评测；脚本中关于数据集路径的注释可按需修改。
- 若需从头构建偏好数据，可先运行 `MMedPO/scripts/run_inference_visual_indirect.sh`；否则直接使用 `MMedPO/data/` 下预生成的数据集进行 DPO 或 SSPO 训练即可。
