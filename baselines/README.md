# baselines — 对比方法

本目录存放与 MMedPO / CaMedPO 进行对比的偏好优化方法实现。

## SimPO

- 论文：*SimPO: Simple Preference Optimization with a Reference-Free Reward*（[arXiv:2405.14734](https://arxiv.org/abs/2405.14734)，PDF 见 `../literature/SimPO Simple Preference Optimization.pdf`）
- 特点：无需参考模型的偏好优化方法，用长度归一化的平均对数概率作为隐式奖励，避免参考模型带来的显存与计算开销。

目录结构：

```
SimPO/
├── alignment/          # 训练与对齐核心代码
├── scripts/            # 训练 / 评测启动脚本
├── training_configs/   # 训练超参配置
├── accelerate_configs/ # 分布式训练配置
├── eval/               # 评测代码
├── on_policy_data_gen/ # on-policy 数据生成
├── generate.py         # 生成脚本
└── environment.yaml    # 环境依赖
```

使用方式见 [`SimPO/README.md`](SimPO/README.md)（官方说明）。

## 添加新的对比方法

1. 在本目录下新建以方法名命名的子目录（如 `baselines/xxx/`）
2. 保留其原始 LICENSE 与 README，并在本文件中补一行说明
