# evaluation — 评测工具

[English version see README_EN.md](README_EN.md)

本目录存放医疗（尤其是放射影像报告方向）常用的自动化评测工具，用于补充 MedEvalKit 之外的专项指标。

| 子目录 | 用途 | 主要指标 |
|--------|------|----------|
| `radgraph/` | 放射报告实体关系图抽取 | F1-RadGraph |
| `CheXbert/` | 胸片报告 14 类观察自动标注 | F1-CheXbert、RadGraph-XL 辅助 |
| `medgemma/` | Google MedGemma 官方 notebooks | 基线模型参考实现 |

## radgraph / RadGraph-XL

将放射报告解析为实体-关系图，用于衡量生成报告与参考报告的事实一致性。

```bash
pip install -e evaluation/radgraph
```

详细用法见 [`radgraph/README.md`](radgraph/README.md)（含 RadGraph-XL 与 F1-RadGraph 计算方式）。

## CheXbert

对胸片报告自动标注 14 类观察（Fracture、Consolidation、Cardiomegaly、Edema、Pleural Effusion 等），论文见 [EMNLP 2020](https://arxiv.org/abs/2004.09167)。

```bash
cd evaluation/CheXbert
pip install -r requirements.txt
# 标注
python src/label.py --reports_path <报告文件> --output_path <输出>
```

> **模型权重需自行下载**：
> - `BiomedVLP-CXR-BERT-general/`（1.7 GB）按官方说明获取
> - `radgraph-xl.tar.gz`（397 MB）获取后解压使用

## medgemma

Google 基于 Gemma 3 的医疗多模态模型，提供 4B / 27B 等变体，含文本与图像理解能力。此处保留官方 notebooks 作为基线模型与调用示例参考，详见 [`medgemma/README.md`](medgemma/README.md)。
