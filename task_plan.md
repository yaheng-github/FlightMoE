# FlightMoE 项目实施计划

## 项目目标

完成 FlightMoE 框架的开发、实验验证与 AAAI 论文投稿。

## 当前状态

- [X] 项目框架设计完成
- [X] 专利交底书撰写完成
- [X] 项目汇报 PPT 完成
- [X] 相关论文调研完成 (24篇)
- [X] 项目书 v1.0 整理完成
- [ ] 代码实现
- [ ] 实验验证
- [ ] 论文撰写

---

## Phase 1: 数据预处理与基线复现 (2-3 周)

**目标**: 建立数据处理流水线，复现核心基线方法

- [X] 1.1 RflyMAD 数据下载与目录结构梳理
- [X] 1.2 ULog/TLog/CSV 解析与 41 维字段提取
- [X] 1.3 数据标准化 (Z-Score) 与滑动窗口切分
- [X] 1.4 STFT 时频图像生成与 RGB 伪彩色编码
- [X] 1.5 飞行阶段标签自动标注 (基于飞行状态机)
- [X] 1.6 复现 MAD-GAN (Burst Expert 基线)
- [X] 1.7 复现 GANomaly (Spectral Expert 基线)
- [X] 1.8 复现 GDN (Consistency Expert 基线)
- [X] 1.9 FP-Growth 关联规则挖掘与邻接矩阵生成

**交付物**:

- `data/preprocessed/` 预处理后的数据集
- `baselines/` 基线模型代码与 checkpoints
- `data/adjacency_matrix.npy` 物理一致性图邻接矩阵

---

## Phase 2: 核心模块开发 (4-5 周)

**目标**: 实现 FlightMoE 四个核心模块

- [ ] 2.1 阶段感知多视图编码器
  - [ ] 时序分支: 1D-CNN / TCN 编码器
  - [ ] 时频分支: 2D-CNN 图像编码器
  - [ ] 阶段嵌入层: 可学习阶段 Token
  - [ ] 交互注意力: 跨模态特征融合
- [ ] 2.2 物理一致性图神经网络
  - [ ] 图构建: 基于 FP-Growth 邻接矩阵
  - [ ] GAT/GCN 层: 特征校准
  - [ ] 一致性残差计算
- [ ] 2.3 稀疏专家路由网络
  - [ ] 路由特征聚合层
  - [ ] 门控网络 (Gating Network)
  - [ ] Top-k 稀疏激活机制
  - [ ] 负载均衡损失
- [ ] 2.4 反事实扰动训练模块
  - [ ] 五类扰动实现 (缺失/异步/噪声/伪影/域偏移)
  - [ ] 复合损失函数 (L_inv + L_consist + L_route)
  - [ ] 对抗训练循环

**交付物**:

- `models/encoder.py` 编码器模块
- `models/gnn.py` 图神经网络模块
- `models/router.py` 路由网络模块
- `models/perturbation.py` 扰动训练模块

---

## Phase 3: 系统集成与训练 (3-4 周)

**目标**: 整合五层架构，完成端到端训练

- [ ] 3.1 五层架构系统集成
- [ ] 3.2 数据加载器 (支持多模态、多阶段、掩码)
- [ ] 3.3 训练脚本 (支持混合精度、分布式)
- [ ] 3.4 验证脚本 (支持开放集评估)
- [ ] 3.5 超参数搜索 (Optuna / WandB Sweeps)
- [ ] 3.6 模型选择与 checkpoint 管理

**交付物**:

- `train.py` 主训练脚本
- `evaluate.py` 评估脚本
- `configs/` 超参数配置文件
- `checkpoints/` 最优模型权重

---

## Phase 4: 实验验证与论文撰写 (4-5 周)

**目标**: 完成实验验证并撰写 AAAI 论文

- [ ] 4.1 SIL/HIL 基准测试
  - [ ] 对比 MAD-GAN, GANomaly, GDN, CMDIAD 等方法
  - [ ] 计算 AUC-ROC, F1-Score, Precision, Recall
- [ ] 4.2 Sim-to-Real 泛化实验
  - [ ] SIL/HIL → Real Flight 跨域测试
  - [ ] 域偏移量化分析
- [ ] 4.3 消融实验
  - [ ] w/o 阶段嵌入
  - [ ] w/o 物理一致性图
  - [ ] w/o 稀疏路由 (平均加权)
  - [ ] w/o 反事实训练
- [ ] 4.4 专家可视化分析
  - [ ] 路由权重热力图
  - [ ] 专家激活模式与飞行阶段关系
- [ ] 4.5 真实无人机验证 (如条件允许)
- [ ] 4.6 AAAI 论文撰写
  - [ ] Introduction + Related Work
  - [ ] Method (四个模块详细描述)
  - [ ] Experiments (主实验 + 消融 + 可视化)
  - [ ] Conclusion
- [ ] 4.7 论文内部评审与修改

**交付物**:

- `experiments/results/` 实验结果表格与图表
- `paper/main.tex` AAAI 论文 LaTeX 源文件
- `paper/figures/` 论文图表

---

## Phase 5: 专利与开源 (并行推进)

- [ ] 5.1 完善专利申请文件 (根据代码实现补充技术细节)
- [ ] 5.2 开源代码仓库整理 (README, LICENSE, requirements)
- [ ] 5.3 数据集处理工具开源
- [ ] 5.4 技术博客撰写

---

## 风险与应对

| 风险                   | 可能性 | 影响 | 应对措施                             |
| ---------------------- | ------ | ---- | ------------------------------------ |
| RflyMAD 数据下载困难   | 中     | 高   | 提前申请数据集权限，准备备用数据集   |
| 真实无人机验证资源不足 | 高     | 中   | 以 SIL/HIL 为主，真实飞行作为加分项  |
| 训练收敛困难           | 中     | 高   | 分阶段预训练，先训练单专家再联合优化 |
| AAAI 截稿时间紧张      | 高     | 高   | 制定详细周计划，每周检查进度         |

---

## 时间线

```text
Week  1-3  [Phase 1] 数据预处理与基线复现
Week  4-8  [Phase 2] 核心模块开发
Week  9-12 [Phase 3] 系统集成与训练
Week 13-17 [Phase 4] 实验验证与论文撰写
Week 18+   [Phase 5] 专利完善与开源
```

*注: Phase 5 与 Phase 4 后半部分并行推进*
