# MMedPO Utils

这个目录包含了用于数据集重建和处理的实用脚本。

## 脚本说明

### 1. recompute_dpo_weights.py
- **功能**: 使用用户提供的compute_dpo_weight函数重新计算DPO权重
- **问题**: 使用per-case标准化，导致权重分布过高（平均0.96）
- **状态**: 已弃用，建议使用improved_dpo_weights.py

### 2. improved_dpo_weights.py ⭐ 推荐使用
- **功能**: 改进的DPO权重计算脚本
- **改进点**:
  - 使用全局标准化而非per-case标准化
  - 调整权重参数以获得更合理的分布
  - 权重分布：平均0.45，范围[0.01, 0.99]
- **使用方法**:
  ```bash
  cd /workspace/MMedPO
  python utils/improved_dpo_weights.py
  ```

## 生成的数据集

### tie_dpo_dataset_improved.jsonl
- **位置**: `/workspace/MMedPO/outputs/tie_dpo_dataset_improved.jsonl`
- **样本数**: 4,919
- **权重分布**: 合理的正态分布，平均0.45
- **格式**: 符合DPO训练要求的JSONL格式

### 权重分布统计
- 总样本数: 4,919
- 权重范围: [0.01, 0.99]
- 平均权重: 0.4540
- 中位数权重: 0.4400
- 权重标准差: 0.2284
- 权重 > 0.5 的样本数: 1,961 (39.9%)
- 权重 > 0.7 的样本数: 729 (14.8%)
- 权重 > 0.9 的样本数: 154 (3.1%)

## 使用建议

1. **训练DPO模型时**，使用 `tie_dpo_dataset_improved.jsonl` 作为数据源
2. **更新训练脚本**中的数据路径：
   ```bash
   --data_path /workspace/MMedPO/outputs/tie_dpo_dataset_improved.jsonl
   ```
3. **权重分布合理**，可以有效进行DPO训练