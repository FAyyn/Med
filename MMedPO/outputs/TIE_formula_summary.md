# TIE 指标与统计量公式汇总（MMedPO）

本文档整理了本项目在 TIE（Token-level/Inference-based Evaluation）计算与输出中使用的主要符号、公式与统计量定义，并给出与 Excel/JSON 输出字段的对应关系，便于对照与复现。

---

## 第一部分：基础概念与符号定义

### 1. 核心术语与符号
- x：文本输入（prompt）
- I：图像输入（可为空）
- y：目标答案（token 序列）
- y⁺、y⁻：正样本答案（preferred/chosen）与负样本答案（rejected）
- T：答案 y 的 token 数；T⁺、T⁻ 分别为 y⁺、y⁻ 的 token 数
- p_θ(·)：参数为 θ 的条件概率分布
- ℓ_t = log p_θ(y_t | y_<t, x, I)：第 t 个 token 的对数条件概率（自然对数，单位为 nats）

### 2. 符号速查表
- Δ⁺、Δ⁻：正/负样本聚合对数似然
- γ（gamma）：样本级差值（Δ⁺ − Δ⁻）
- m_v：γ 的别名
- ȳ⁺、ȳ⁻：正/负样本 token 级平均对数似然
- m_n：token 级差值（ȳ⁺ − ȳ⁻）
- S_disp：−min(0, γ)（方案 B）

---

## 第二部分：核心计算公式

### 3. 对数似然与 token 级度量
1) **逐 token 对数似然（Teacher Forcing）**
- 定义：ℓ_t = log p_θ(y_t | y_<t, x, I)
- 样本 y 的总对数似然：
  - LL(y | x, I) = Σ_{t=1}^{T} ℓ_t
- token 级平均对数似然（长度归一化）：
  - ȳ = (1/T) Σ_{t=1}^{T} ℓ_t

2) **正/负样本的 token 级均值**
- 对正样本 y⁺：
  - ȳ⁺ = (1/T⁺) Σ_{t=1}^{T⁺} ℓ_t^(⁺)
- 对负样本 y⁻：
  - ȳ⁻ = (1/T⁻) Σ_{t=1}^{T⁻} ℓ_t^(⁻)

### 4. 样本级 TIE 指标（聚合量）
1) **正/负样本聚合分数**
- Δ⁺ = LL(y⁺ | x, I) = Σ_{t=1}^{T⁺} ℓ_t^(⁺)
- Δ⁻ = LL(y⁻ | x, I) = Σ_{t=1}^{T⁻} ℓ_t^(⁻)

2) **样本级差值 γ（也记为 m_v）**
- 定义：γ = Δ⁺ − Δ⁻

3) **token 级归一化差值 m_n**
- 定义：m_n = ȳ⁺ − ȳ⁻ = (1/T⁺) Σ_{t=1}^{T⁺} ℓ_t^(⁺) − (1/T⁻) Σ_{t=1}^{T⁻} ℓ_t^(⁻)

### 5. 离散度度量 S_disp（方案 B）
- 采用定义：S_disp = −min(0, γ)
- 分段等价形式：
  - 当 γ ≥ 0：S_disp = 0
  - 当 γ < 0：S_disp = −γ
- 直观解释：强调"反向强度"。当模型更偏好负样本（γ 为负）时，S_disp 为正并等于偏好差的绝对值；当模型偏好正样本时，S_disp 为 0。

---

## 第三部分：DPO 权重计算

### 6. DPO 权重计算公式（基于 TIE 特征）

本节提供一组**一步到位的公式**，将已有的 TIE 特征（token-normalized）转换成用于 DPO 的权重 $w$（或偏好概率 $p$）。公式按顺序：聚合 → 归一化 → 映射 → 变换 → 裁剪。

#### 6.1 记号与输入
- $\Delta^+, \Delta^-$：token-normalized TIE for positive / negative answers
- $\gamma = \Delta^+ - \Delta^-$
- $m_v = \mathrm{LL}_{\text{pos}}(bg+full) - \mathrm{LL}_{\text{neg}}(bg+full)$
- $m_n = \mathrm{LL}_{\text{pos}}(bg+white) - \mathrm{LL}_{\text{neg}}(bg+white)$
- （可选）$\delta = \Delta^+_{\text{obj}} - \Delta^-_{\text{obj}}$（object-only 差分）

#### 6.2 参数（推荐初始值）
- $\alpha = 0.5, \beta = 0.5, \eta = 0.3$
- $\tau_n = 80\%$ 分位的 $m_n$
- 归一化 clip 上限 $K = 10$
- 温度 $\beta_T = 1.0$
- 幂缩放 $\kappa = 0.5$
- 权重裁剪 $[w_{\min}, w_{\max}] = [0.01, 10]$
- flip 阈值 $\tau_{\text{flip}} = 0.3$

#### 6.3 合成偏好原始分 $S$

$$
S_{\text{raw}} = \gamma + \alpha \, m_v - \beta \cdot \max(0, m_n - \tau_n) + \eta \, \delta
$$

（若无 $\delta$ 或 $\delta$ 不可用，省略该项。）

#### 6.4 批内标准化

在一个 batch 或窗口内计算均值与标准差：

$$
\mu = \mathrm{mean}(S_{\text{raw}}), \quad \sigma = \mathrm{std}(S_{\text{raw}}) + \varepsilon
$$

标准化并裁剪：

$$
S' = \operatorname{clip}\left(\frac{S_{\text{raw}} - \mu}{\sigma}, -K, K\right)
$$

#### 6.5 映射到偏好概率 $p$

$$
p = \sigma(\beta_T \cdot S') = \frac{1}{1 + e^{-\beta_T S'}}
$$

为数值稳定，强制 $p \in [\varepsilon, 1-\varepsilon]$，通常 $\varepsilon = 10^{-6}$。

#### 6.6 变换成训练权重 $w$

**策略一（直接幂缩放，推荐）**：

$$
w = \operatorname{clip}(p^{\kappa}, w_{\min}, w_{\max})
$$

**策略二（中心化后线性放大）**：
先中心化：$s = 2p - 1 \in (-1, 1)$，再线性映射到正权重：

$$
w = \operatorname{clip}(a \cdot s + b, w_{\min}, w_{\max})
$$

（推荐使用幂缩放，$\kappa = 0.5$ 即 $w = \sqrt{p}$）

#### 6.7 处理强反偏好（flip）

若该样本满足"前景更支持负答"的情况：

$$
\text{is\_flip} \Longleftrightarrow \Delta^- - \Delta^+ > \tau_{\text{flip}}
$$

对 flip 样本有两种等效做法（任选其一）：

- **A) 翻转对序**：把 pair 的正负顺序翻转（使 DPO 学习"偏好负答"），并用较高的权重 $w_{\text{flip}} = \rho \cdot w$（例如 $\rho = 1.5$）
- **B) 保持序列但放大权重**：直接把该样本 $w \gets \rho \cdot w$ 并保留原 order，同时在 loss 中把它作为强惩罚

#### 6.8 在 DPO 中的使用（pairwise loss）

令模型对该 pair 预测的"pos 被偏好"的概率为 $p_{\text{model}}$。常用的有加权对数损失：

$$
\mathcal{L}_{\text{pair}} = -w \cdot \log(p_{\text{model}})
$$

（若翻转了对序，上式仍适用；若希望 soft-target，可把 $p$ 用作目标值并最小化加权交叉熵：$-w[p\log p_{\text{model}} + (1-p)\log(1-p_{\text{model}})]$）

#### 6.9 完整权重计算公式（单行表达式）

**完整版本**（含所有辅助项）：

$$
w = \operatorname{clip}\left( \left[ \sigma\left( \beta_T \cdot \operatorname{clip}\left( \frac{ \Delta^+ - \Delta^- + \alpha m_v - \beta \max(0, m_n - \tau_n) + \eta \delta - \mu }{\sigma}, -K, K \right) \right) \right]^{\kappa}, w_{\min}, w_{\max} \right)
$$

**简化版本**（仅保留 $\Delta^+ - \Delta^-$）：

$$
w = \operatorname{clip}\left( \left[ \sigma\left(\beta_T \cdot \frac{ \Delta^+ - \Delta^- - \mu}{\sigma}\right) \right]^{\kappa}, w_{\min}, w_{\max} \right)
$$

#### 6.10 总结性流程

给定样本的最终权重计算流程：

1. **合成原始分**：$S_{\text{raw}} = \Delta^+ - \Delta^- + \alpha m_v - \beta\max(0, m_n - \tau_n) + \eta\delta$
2. **标准化**：$S' = \operatorname{clip}((S_{\text{raw}} - \mu)/\sigma, -K, K)$
3. **Sigmoid 映射**：$p = \sigma(\beta_T S')$
4. **幂缩放**：$w = \operatorname{clip}(p^{\kappa}, w_{\min}, w_{\max})$
5. **Flip 处理**：若 $\Delta^- - \Delta^+ > \tau_{\text{flip}}$ 则 $w \gets \rho \cdot w$ 或翻转 pair

---

## 第四部分：统计分析与汇总

### 7. Summary（总体统计，Excel Summary 工作表与 JSON.summary）
对每个样本得到 m_n、γ、ȳ⁺、ȳ⁻、S_disp 后，进行如下总体统计：

- **计数**
  - count_m_n = 有效 m_n 的样本数 N

- **均值**
  - mean_m_n_token_level = (1/N) Σ_{i=1}^{N} m_n^{(i)}
  - mean_gamma = (1/N) Σ_{i=1}^{N} γ^{(i)}
  - mean_S_disp = (1/N) Σ_{i=1}^{N} S_disp^{(i)}
  - mean_tie_positive_token_avg = (1/N) Σ_{i=1}^{N} ȳ⁺^{(i)}
  - mean_tie_negative_token_avg = (1/N) Σ_{i=1}^{N} ȳ⁻^{(i)}

- **样本方差与标准差**（针对 m_n，ddof = 1）
  - var_m_n_token_level = (1/(N − 1)) Σ_{i=1}^{N} [m_n^{(i)} − mean_m_n_token_level]^2
  - std_m_n_token_level = sqrt(var_m_n_token_level)

注：若后续需要，可对 γ、ȳ⁺、ȳ⁻、S_disp 等也计算方差、分位数等统计量。

---

## 第五部分：数据结构与字段对照

### 8. 输出字段与含义对照
- **TIE 核心指标**
  - tie_positive：Δ⁺（正样本聚合对数似然）
  - tie_negative：Δ⁻（负样本聚合对数似然）
  - tie_difference、gamma：γ = Δ⁺ − Δ⁻
  - m_v：γ 的别名（与 gamma 等价）
  - tie_positive_token_avg：ȳ⁺（正样本 token 平均对数似然）
  - tie_negative_token_avg：ȳ⁻（负样本 token 平均对数似然）
  - m_n：ȳ⁺ − ȳ⁻（token 级差值）
  - S_disp：−min(0, γ)（方案 B）

- **权重与元数据**
  - weighted_score：DPO权重 w（基于TIE特征计算得出）
  - answer_type：来自数据集（如 OPEN/YESNO/MULTIPLE_CHOICE 等）

### 9. 正负样本对与权重形式

#### 9.1 原论文中的样本对与权重
- 文本对比对：{ (Y_{gt} | X), (Y_{gen} | X) }，用于衡量模型对真实答案与幻觉生成（hallucination）的区分能力。
- 输入对比对：{ (Y_{gt} | X), (Y_{gt} | X′) }，用于衡量同一答案在原始输入与修改输入（如遮挡/替换/扰动）下的稳健性。
- 权重来源：由 visual grounding 与 multi-agent 评审分别给出置信度/评分，并在原论文中融合后得到每个样本对的权重 w；具体融合细节以原论文为准。

#### 9.2 我们的方法中的样本对与权重
- **数据来源与正负样本**：
  - DPO 格式：positive_answer = 接受答案 Y⁺；negative_answer = 拒绝答案 Y⁻；图像路径由 image 提供（脚本在该格式下将 masked 与 full 视作同一路径）。
  - extracted_where_questions 格式：给出 full_image_path 与 masked_image_path；同时提供 positive_answer 与 negative_answer。
- **TIE 中的样本对**（背景处理的一致性对照）：
  - 对正样本 Y⁺：构造两种图像上下文 X_mask 与 X_white，并形成对 { (Y⁺ | X_mask), (Y⁺ | X_white) }，其差值定义为 TIE_positive = LL(Y⁺ | X_mask) − LL(Y⁺ | X_white)。
  - 对负样本 Y⁻：同理形成对 { (Y⁻ | X_mask), (Y⁻ | X_white) }，TIE_negative = LL(Y⁻ | X_mask) − LL(Y⁻ | X_white)。
  - 合成差值：γ = TIE_positive − TIE_negative；同时给出 token 归一化指标 m_n 以及 ȳ⁺、ȳ⁻（详见前文第 3–4 节）。

#### 9.3 weighted_score（DPO权重 w）
- **定义**：`weighted_score` 现为基于TIE特征计算的DPO权重 $w$，不再从输入数据直接读取。
- **计算流程**：
  1. 合成原始分：$S_{\text{raw}} = \Delta^+ - \Delta^- + \alpha m_v - \beta\max(0, m_n - \tau_n) + \eta\delta$
  2. 批内标准化：$S' = \operatorname{clip}((S_{\text{raw}} - \mu)/\sigma, -K, K)$
  3. Sigmoid映射：$p = \sigma(\beta_T \cdot S')$
  4. 幂缩放：$w = \operatorname{clip}(p^{\kappa}, w_{\min}, w_{\max})$
- **处理**：计算后通过 `weighted_score = round(w, 4)` 进行小数点后四位四舍五入，确保数值精度。
- **用途**：作为DPO训练中的样本权重，直接参与损失函数计算，影响模型偏好学习效果。

#### 9.4 与现有字段的对应关系

- $\Delta^+$ ↔ `tie_positive_token_avg`（ȳ⁺）
- $\Delta^-$ ↔ `tie_negative_token_avg`（ȳ⁻）
- $\gamma$ ↔ `tie_difference` 或 `gamma`
- $m_n$ ↔ `m_n`
- $m_v$ ↔ `m_v`
- 最终权重 $w$ → 新增字段 `dpo_weight`

### 10. TIE 相关参数计算方式
- **tie_positive**：`ll_pos_mask - ll_pos_white`，即正样本在不同背景下的 log likelihood 差值。
- **tie_negative**：`ll_neg_mask - ll_neg_white`，即负样本在不同背景下的 log likelihood 差值。
- **tie_difference**：`tie_positive - tie_negative`，衡量正负样本 TIE 效应的差异。
- **token 长度归一化**：
    - `tie_pos_token_avg = tie_positive / max(1, pos_token_len)`
    - `tie_neg_token_avg = tie_negative / max(1, neg_token_len)`
- **gamma_value**：等同于 `tie_difference`，用于后续新指标计算。
- **m_n_value**：`tie_pos_token_avg - tie_neg_token_avg`，反映 token 归一化后正负样本 TIE 差异。
- **S_disp_value**：`-min(0, gamma_value)`，用于区分 gamma 为负时的离散度。

### 11. 字段归属与输出结构
- 以上参数均在 `result` 字典中输出，便于后续保存为 csv/excel 或分布式收集。
- 相关字段包括：`weighted_score`（现为计算得出的DPO权重w）、`tie_positive`、`tie_negative`、`tie_difference`、`gamma`、`m_n`、`S_disp`、token 长度、模型输出等。

---

## 第六部分：实现细节与技术说明

### 12. JSON 输出说明
- 结果文件从 JSONL 切换为 JSON 聚合对象，顶层结构为：
  - results：逐样本的完整结果（含上述新增字段）
  - summary：与 Excel Summary 工作表一致的总体统计

### 13. 实现与数值注意事项
- **对数底**：默认自然对数（nats）。若需以 bits 表示，可统一换底为 log₂。
- **长度归一化**：m_n 使用 token 平均，缓解不同答案长度导致的偏差；γ 使用聚合量，反映总似然偏好。
- **数据字段一致性**：
  - gamma ≡ tie_difference ≡ m_v
  - m_n 由 ȳ⁺ 与 ȳ⁻ 直接计算
  - S_disp 采用方案 B 定义
- **数据来源**：answer_type 等元数据来自原始样本（如 Slake1.0 的 master_question_with_dpo.json）。
- **异常处理**：推理异常时各参数置零或填充为 "ERROR"，保证数据完整性。
- **分布式收集**：多卡训练时通过 `dist.all_gather_object` 合并各卡结果，去重并排序后统一保存。
- **归一化与取整**：所有权重与 TIE 指标均保证数值一致性，便于下游分析与论文复现。

---

（完）