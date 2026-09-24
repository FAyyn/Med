# analysis — 实验结果分析与可视化

[English version see README_EN.md](README_EN.md)

本目录（`analysis/`）存放实验结果分析脚本，用于对比 **baseline / MMedPO / CaMedPO（CaMPPO）** 三种方法在医学 VQA 与报告生成任务上的表现，并按类别整理、可视化典型案例。

## 脚本分类

### 1. 结果对比分析

| 脚本 | 用途 |
|------|------|
| `analyze_medical_results.py` | 主分析脚本：对比三种方法在 VQA 与报告生成上的性能 |
| `analyze_pathological_cases.py` | 针对病理类（pathological）案例的专项分析 |
| `enhance_report_cases.py` | 补充 / 增强报告生成案例的分析维度 |

### 2. 案例分类与匹配

| 脚本 | 用途 |
|------|------|
| `categorize_cases.py` | 按 `open`（开放式）、`close`（封闭式）、`report`（报告生成）三类整理案例 |
| `match_cases_with_images.py` | 根据问题匹配对应的影像文件 |
| `filter_camppo_superior_cases.py` | 筛选 CaMedPO 明显优于对比方法的案例 |

### 3. 可视化与报告生成

| 脚本 | 用途 |
|------|------|
| `create_camppo_best_visualization.py` | 生成 CaMedPO 最优案例可视化页面 |
| `create_camppo_best_cases_report.py` | 生成最优案例报告 |
| `create_pathological_visualization.py` | 病理案例可视化 |
| `create_enhanced_pathological_cases.py` | 增强版病理案例整理 |
| `create_visual_examples.py` | 通用可视化样例生成 |
| `create_detailed_mapping_report.py` | 生成详细映射报告 |
| `create_enhanced_mapping.py` | 增强映射关系构建 |
| `update_case_image_summary.py` | 更新案例图片汇总 |
| `update_report_with_images.py` | 将图片回填到报告中 |

## 输出产物

脚本运行后会在同目录生成分析结果：

- `close_vqa_cases.json`、`open_vqa_cases.json` — 按类型拆分的 VQA 案例
- `camppo_superior_cases.json` + `camppo_superior_analysis.md` — CaMedPO 优势案例与分析
- `enhanced_pathological_cases.json` + `enhanced_pathological_cases_analysis.md` — 病理案例分析
- `pathological_cases_analysis.json` / `.md`
- `pathological_cases_visualization.html` — 可直接在浏览器打开的可视化页面
- `summary_by_category.json`、`report_generation_cases.json` — 分类汇总与报告生成案例

## 运行方式

脚本均为独立的 Python 脚本，按需修改脚本内的数据路径后直接运行：

```bash
python analysis/analyze_medical_results.py
python analysis/categorize_cases.py
python analysis/create_camppo_best_visualization.py
```

> 脚本读取的是 MMedPO / MedEvalKit 的评测输出，运行前请先完成 [`../MedEvalKit`](../MedEvalKit/Readme.md) 的评测流程。
