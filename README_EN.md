# Med — Medical Vision-Language Model Preference Optimization

This repository contains the code, evaluation tools, experiment analysis and documentation for **MMedPO** (clinical-aware multimodal preference optimization) and its follow-up work **CaMedPO / CaMPPO** (causality-aware preference optimization).

- Main project paper: MMedPO — [arXiv:2412.06141](https://arxiv.org/abs/2412.06141)
- Follow-up paper: *Causality-Aware Preference Optimization for Aligning Medical Vision Language Models*
- Evaluation framework: MedEvalKit — [arXiv:2506.07044](https://arxiv.org/abs/2506.07044)

[中文版说明见 README.md](README.md)

---

## Repository structure

| Directory | Description |
|-----------|-------------|
| [`MMedPO/`](MMedPO/README_EN.md) | Main project: data curation, SFT / DPO / SSPO / GRPO training, inference, evaluation and utility scripts |
| [`MedEvalKit/`](MedEvalKit/Readme.md) | Unified medical multimodal evaluation framework |
| [`baselines/`](baselines/README_EN.md) | Baseline methods (SimPO etc.) |
| [`evaluation/`](evaluation/README_EN.md) | Evaluation tools: RadGraph, CheXbert, MedGemma |
| [`analysis/`](analysis/README_EN.md) | Experiment analysis, case categorization and visualization scripts |
| [`literature/`](literature/README_EN.md) | Reference papers (PDF) |

---

## Quick start

### 1. Environment

```bash
# Main project (MMedPO)
conda create -n MMedPO python=3.10 -y
conda activate MMedPO
pip install --upgrade pip
pip install -r MMedPO/requirements.txt
pip install trl

# Evaluation framework (MedEvalKit)
pip install -r MedEvalKit/requirements.txt
```

Model weights must be downloaded separately: base model [LLaVA-Med-1.5](https://huggingface.co/microsoft/llava-med-v1.5-mistral-7b), official MMedPO checkpoints at [mmedpo_checkpoints](https://huggingface.co/zky11235/mmedpo_checkpoints).

### 2. Data

- **SLAKE dataset** and processed occluded background images: [Google Drive](https://drive.google.com/file/d/1YJC7KspZohlfGxylWAKc6bVxfMZKGbYS/view?usp=drive_link)
- Pre-built preference data under `MMedPO/data/`, ready to use:
  - `slake_dpo_weighted.json` — weighted DPO data for SLAKE
  - `tie_dpo_dataset_method1_converted.json` — DPO data built with method 1 (tie weights)
  - `iuxray_dpo_weighted.json`, `mimic_dpo_weighted.json` — IU-Xray / MIMIC data
  - `iuxray_sft_dataset.json` — SFT-format data

### 3. Training

All training scripts are in `MMedPO/scripts/`. Edit parameters (data path, model path, GPU allocation, etc.) directly in the script, then run `bash MMedPO/scripts/xxx.sh`.

| Script | Description |
|--------|-------------|
| `train_sft.sh` | SFT training; can directly use DPO data converted to SFT format |
| `train_dpo.sh` | DPO training |
| `train_dpo_visual-text.sh` / `train_dpo_visual-text_optimized.sh` | Visual-text DPO training (the latter is the optimized version) |
| `train_sspo.sh` | Standard SSPO |
| `train_sspo_adv.sh` | SSPO: uses weights instead of dynamic w in the loss |
| `train_tie_sspo.sh` | SSPO: with dynamic w computation |
| `run_sppo.py` | SSPO training entry point |
| `MMedPO/train/rl/train_grpo_stage3.py` | GRPO reinforcement learning stage |

### 4. Preference pair construction and inference

| Script | Description |
|--------|-------------|
| `run_inference_visual_indirect.sh` | Method 1: takes `slake_dpo_weighted.json`, computes tie weights and builds preference pairs |
| `run_inference_text_contrast.sh` | Text-contrast inference |
| `run_inference_visual_consistency.sh` | Visual-consistency inference |
| `run_full_inference_background.sh` | Full inference with randomized background |
| `inference_llava-med_vqa.sh` / `inference_llava-med_report.sh` | LLaVA-Med VQA / report generation inference |
| `inference_attention-map_score.sh` | Attention-map score inference |

### 5. Evaluation

- **MedEvalKit**: `MedEvalKit/eval.py` with `eval.sh` / `eval_chunked.sh`; see [`MedEvalKit/Readme.md`](MedEvalKit/Readme.md)
- **SLAKE and LLM-as-judge**: `eval_slake_inference.py`, `eval_slake_iou.py`, `eval_slake_combined.py`, `run_evaluate_llmjudge.sh`
- **IoU computation**: `run_compute_iou_slake.sh` (with `compute_iou_slake.py`)
- **Metric tools**: RadGraph and CheXbert under `evaluation/`; see [`evaluation/README_EN.md`](evaluation/README_EN.md)

### 6. Result analysis

Scripts under `analysis/` compare baseline / MMedPO / CaMedPO, organize cases by category and generate visualization pages; see [`analysis/README_EN.md`](analysis/README_EN.md).

---

## Notes

- The evaluation scripts support checkpoints, so you can evaluate a model directly from a checkpoint; comments about dataset paths inside the scripts can be modified as needed.
- To build preference data from scratch, run `MMedPO/scripts/run_inference_visual_indirect.sh` first; otherwise simply use the pre-built datasets under `MMedPO/data/` for DPO or SSPO training.
