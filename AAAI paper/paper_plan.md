# FlightMoE v2 AAAI 投稿论文规划

> 本文件用于确定论文结构、AAAI 格式要求、以及哪些结果放入正文/补充材料。
> 先在此文档中达成一致，再往 `main.tex` 里写。

---

## 一、AAAI 投稿基本要求

参考当前 `AAAI paper/` 中的 `aaai.sty` 和 `formatting-instructions-latex.tex`：

| 项目 | 要求 |
|---|---|
| 页数 | 正文 7 页 + 引用 1 页（共 8 页）* |
| 格式 | 双栏、US letter |
| 字体 | Times + Helvetica + Courier |
| 匿名 | 提交版必须匿名，不能出现作者、单位、致谢 |
| 文件 | 单 `.tex` 文件（不能用 `\input{}`） |
| 图表 | 使用 `figure`/`table`，图片建议 300 dpi 以上；图表数量无硬性上限，但占用正文页数 |
| 参考文献 | 使用 `aaai.bst`，放在正文后第 8 页；数量无硬性上限，建议 10–15 条 |
| 字数 | 无硬性字数限制，由页数间接约束；摘要通常建议 150–200 词 |

\* 具体页数请以当年 AAAI Call for Papers 为准。

**当前问题**：
- `main.tex` 目前约 8-9 页内容，**需要压缩到 7 页正文**
- 表格和图片过多，需要筛选
- 部分表述更像技术报告，不是学术论文风格

---

## 二、建议论文结构（7 页正文）

### 第 1 页：Title + Abstract + Introduction

**Abstract（约 150-200 词）**
- 问题：UAV 异常检测的重要性和挑战
- 方法：一句话概括 FlightMoE v2（phase-aware + multimodal + sparse MoE）
- 主要结果：closed/open AUC/F1
- 关键结论：spectral/phase/sparse routing 的重要性

**Introduction（约 0.8-1 页）**
1. 背景：UAV 安全关键任务，传感器故障危害大
2. 现有方法问题：
   - 把飞行当成平稳过程，忽略 phase 非平稳性
   - 单一 scorer 无法覆盖多种故障模式
3. 我们的方法：
   - Phase-aware multimodal encoder
   - Sparse expert router（Top-k）
   - Physical-consistency GNN
   - Margin ranking loss for unsupervised pre-training
4. 贡献（用 bullet，3-4 条）
5. 主要结果预览

**本部分需要的图/表**：
- 可以放一个 **小型的 motivation 图**（可选，非必须）
- 不放详细架构图（放到 Method）

---

### 第 2 页：Related Work

**内容：**
1. UAV anomaly detection（MAD-GAN / GANomaly / GDN / HMM）
   - 指出它们 phase-agnostic、single-scorer 的局限
2. Mixture of Experts in anomaly detection
   - 前人做 score-level fusion，我们做 feature-level sparse routing
3. Multimodal time-series anomaly detection
   - STFT/spectral features的作用

**本部分不放图/表。**

---

### 第 2-3 页：Method

**建议结构：**

#### 3.1 Problem Formulation
- 输入：$X_t, S_t, p_t$
- 输出：$a_t$
- 三阶段训练概述

#### 3.2 Phase-Aware Multi-View Encoder
- Temporal branch
- Spectral branch
- Cross-modal attention
- Phase embedding

#### 3.3 Physical Consistency GNN
- 输入 encoder 输出，投影到 per-sensor
- 阶段特定邻接矩阵
- GAT 传播
- Consistency residual

#### 3.4 Sparse Expert Router
- Router 输入
- Top-k gating
- Load-balancing loss

#### 3.5 Expert Heads and Final Scoring
- 4 个 expert
- 最终分数 = AnomalyHead(z) + Σ w_i s_i

#### 3.6 Training Strategy
- Stage 1：ranking loss
- Stage 2：train router
- Stage 3：joint fine-tuning

**本部分需要的图：**
- **Figure 1: 架构图**（放在 3.1 或 3.2 附近）
  - 展示：输入 → Encoder → GNN → Router → Experts → Score
  - 当前 `figures/architecture.tex` 可以，但建议画得更清晰

**本部分需要的公式：**
- Cross-attention（可选）
- Top-k gating
- Load-balancing loss
- Margin ranking loss
- 最终分数

---

### 第 4 页：Experiments - Dataset & Main Results

#### 4.1 Dataset and Setup
- RflyMAD 简介
- 41 sensors / 6 phases / train/val/test_closed/test_open split
- Open set 包含 unseen sensor faults
- 实现细节：GPU、batch size、optimizer、epochs

#### 4.2 Main Results
- **Table 1**: Main results（只有 FlightMoE v2 一行）
  - val / closed / open 的 AUC + F1
- 紧接 Table 1 下方写分析：结果整体水平、closed vs open 的差距说明什么

---

### 第 5 页：Experiments - Baselines & Ablations

#### 4.3 Comparison with Baselines
- **Table 2**: vs FlightMoE v1 / GDN-phase / GDN-full
- 紧接 Table 2 下方分析：v2 为什么比 score-level router 和单专家 GDN 好

#### 4.4 Ablation Study
- **Table 3**: w/o Phase / w/o GNN / w/o Sparse Router / w/o Spectral
- 紧接 Table 3 下方逐项分析：
  - Phase embedding 的重要性
  - Spectral branch 对 open set 的关键作用
  - Sparse routing vs dense
  - GNN 贡献有限的原因

#### 4.5 Effect of Top-k Sparse Routing
- **Table 4**: k=1,2,3,4 的结果
- 紧接 Table 4 下方分析：为什么 k=2 最优

**本部分图：**
- 不放图，只放表（节省空间）

---

### 第 6 页：Experiments - Visualization

#### 4.6 Visualization and Analysis
- **Figure 2**: Router 热力图（test_closed）
- 紧接 Figure 2 下方分析：
  - 不同 phase 激活不同 expert
  - 哪个 phase 用哪个 expert 多，为什么

**可选内容（如果空间允许）：**
- 在 Router 热力图分析后，可以补充 1-2 个关键观察，但避免新增表格

---

### 第 7 页：Conclusion + Limitations

#### Conclusion
- 总结方法
- 强调 open-set 提升
- Future work

#### Limitations（强烈推荐加，AAAI 现在重视）
- GNN 贡献有限，需要更强的物理一致性建模
- 只在 RflyMAD 上验证
- 在线/streaming 检测未探索

---

### 第 8 页：References

- 使用 `aaai.bst`
- 当前 6 条引用太少，建议补充到 10-15 条
- 需要补充：
  - 更多 UAV anomaly detection 相关工作
  - MoE 相关工作
  - 时频分析相关工作
  - Transformer/attention for time series（如果提到 cross-attention）

---

## 三、结果筛选建议

**排版原则：每个图/表下方紧跟 2-4 句分析**，不要先堆砌所有表再统一分析。

### 必须放入正文的结果

| 内容 | 形式 | 理由 |
|---|---|---|
| Main results | Table 1 | 核心结果，表后紧跟分析 |
| Baseline comparison | Table 2 | 证明超越 SOTA，表后紧跟分析 |
| Ablation (phase/GNN/sparse/spectral) | Table 3 | 验证模块贡献，表后逐项分析 |
| Top-k ablation | Table 4 | 验证稀疏路由设计，表后分析 k=2 最优原因 |
| Router heatmap | Figure 2 | 可视化专家激活模式，图后分析 phase-expert 关系 |

### 建议放入 Supplementary 的结果

| 内容 | 理由 |
|---|---|
| Per-phase metrics | 太细，正文放不下 |
| Per-fault-type metrics | 太细，正文放不下 |
| Training curves | 可选 |
| GNN Refine 结果 | 效果不好，不值得占空间 |
| GNN-to-head 结果 | 效果不好，不值得占空间 |

### 是否需要补的实验

1. **多随机种子**（最高优先级）
   - 当前结果是单 seed，审稿人可能质疑
   - 建议跑 3 个 seed，报告 mean ± std

2. **BCE vs Ranking loss 严格消融**
   - 同一 seed 下对比 Stage 1 损失函数
   - 目前在默认配置里 ranking 是默认，可以做一个对比表

3. **GNN 改进**
   - 可学习边权重正在跑
   - 如果效果好，可以替换/补充 w/o GNN 的分析

---

## 四、当前论文需要修改的问题

1. **页数超标**：需要删除部分表格和冗余文字
2. **Introduction 不够聚焦**：需要更明确的 problem-motivation-contribution 三段式
3. **Related Work 太短**：需要扩展，明确我们的区别
4. **Method 公式太多/太少**：需要平衡，保留最核心的
5. **Figure 质量**：架构图需要更清晰，router 热力图保留
6. **Limitations 缺失**：建议补充
7. **引用不够**：需要扩展到 10-15 条

---

## 五、下一步行动

请确认或修改以下内容：

- [ ] 论文结构是否按上述 7 页规划？
- [ ] 哪些表保留在正文，哪些放 supplementary？
- [ ] 是否需要跑多随机种子实验？
- [ ] 是否保留 Limitations 章节？
- [ ] 是否同意当前主结果采用 Stage1 Ranking（0.9973/0.9773/0.9500）？
- [ ] 是否需要重新画架构图？

---

## 六、待填入的核心数字

| Metric | Value |
|---|---|
| Val AUC | 0.9973 |
| Val F1 | 0.9800 |
| Closed AUC | 0.9773 |
| Closed F1 | 0.9539 |
| Open AUC | 0.9500 |
| Open F1 | 0.9197 |

**Baseline 对比数字**：

| Model | Closed AUC | Open AUC |
|---|---|---|
| FlightMoE v1 | 0.9611 | 0.9138 |
| GDN (phase) | 0.9491 | 0.8959 |
| GDN (full) | 0.9601 | 0.9032 |
| FlightMoE v2 | 0.9773 | 0.9500 |

**消融数字**：

| Model | Closed AUC | Open AUC |
|---|---|---|
| Full | 0.9773 | 0.9500 |
| w/o Phase | 0.9589 | 0.9258 |
| w/o GNN | 0.9764 | 0.9458 |
| w/o Sparse Router | 0.9731 | 0.9366 |
| w/o Spectral | 0.9582 | 0.8939 |

**Top-k 消融数字**：

| k | Closed AUC | Open AUC |
|---|---|---|
| 1 | 0.9616 | 0.8767 |
| 2 | 0.9773 | 0.9500 |
| 3 | 0.9261 | 0.7778 |
| 4 | 0.9435 | 0.8700 |
