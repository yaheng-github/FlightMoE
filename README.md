# FlightMoE: Phase-Aware and Reliability-Guided Multimodal Expert Routing for Open-Set UAV Anomaly Detection

基于 **Mixture of Experts (MoE)** 架构的无人机多模态异常检测框架，针对 **RflyMAD** 数据集，目标投稿至 **AAAI**。

---

## 项目简介

FlightMoE 是一个面向无人机（UAV）开放集异常检测的研究框架。核心思想是根据飞行阶段、信号动态和模态完整性，动态激活并融合四类互补的异常检测专家：

- **Burst Expert**：检测时序突变、脉冲干扰
- **Drift Expert**：检测缓慢漂移、统计分布偏移
- **Spectral Expert**：检测频域结构异常、振动模式畸变
- **Consistency Expert**：检测传感器间物理一致性断裂

当前版本（v1）已实现 **Score-Level MoE Router**：冻结四类专家，训练 MLP Router 对专家异常分数进行动态融合。

---

## 数据集与数据处理

本项目使用 **RflyMAD** 数据集进行实验验证：

- 数据集主页：[RflyMAD - A Dataset for Multicopter Fault Detection and Health Assessment](https://rfly-openha.github.io/documents/4_resources/dataset.html#rflymad-a-dataset-for-multicopter-fault-detection-and-health-assessment)
- 数据规模：约 114 GB，5,629 个 flight cases
- 数据类型：SIL（软件在环）、HIL（硬件在环）、Real Flight（真实飞行）
- 传感器维度：41 维 UAV 传感器 telemetry
- 飞行阶段：hover / waypoint / velocity / circling / acce / dece

数据预处理采用官方提供的数据处理工具：

- [lerlis/Data_processing_tools](https://github.com/lerlis/Data_processing_tools.git)

本地仓库中 `Data_processing_tools/` 为该工具的本地副本，不参与本仓库版本控制。

---

## 目录结构

```text
FlightMoE/
├── data/
│   ├── preprocessed/          # 预处理后的数据（大文件已排除）
│   │   ├── train.npz          # 训练集 [N, 128, 41]
│   │   ├── val.npz            # 验证集
│   │   ├── test_closed.npz    # 闭集测试集
│   │   ├── test_open.npz      # 开集测试集
│   │   └── split_metadata.json
│   └── adjacency/             # 6 个阶段物理一致性邻接矩阵
│       ├── adjacency_{phase}.npy
│       └── rules_{phase}.json
├── src/
│   ├── data/                  # 数据预处理脚本
│   │   ├── preprocess.py
│   │   ├── split.py
│   │   ├── stft_generator.py
│   │   ├── build_adjacency_matrix.py
│   │   └── visualize_spectral.py
│   ├── baselines/             # 四类专家实现（基于第三方方法修改）
│   │   ├── mad_gan/           # Burst Expert（基于 MAD-GAN 修改）
│   │   ├── ganomaly/          # Spectral Expert（基于 GANomaly 修改）
│   │   ├── gdn/               # Consistency Expert（基于 GDN 修改）
│   │   └── hmm_drift/         # Drift Expert（基于 HMM 思想实现）
│   ├── models/
│   │   └── router_v1.py       # FlightMoE v1 Score-Level Router
│   └── utils/
│       └── experiment_utils.py
├── experiments/               # 实验结果与导出分数
│   ├── scores/
│   ├── gdn_ablation_fast/
│   ├── router_v1/
│   └── paper/
├── docs/                      # 文档
│   ├── FlightMoE_Project_Proposal.md
│   ├── baseline_results.md
│   ├── current_status_and_roadmap.md
│   ├── data_pipeline.md
│   └── Dataset_Fields_Reference.md
├── task_plan.md               # 项目实施计划
├── progress.md                # 项目进度日志
└── README.md                  # 本文件
```

---

## 基线专家与参考论文

本项目中的专家模型均基于已有的经典异常检测方法进行修改与适配：

| 专家 | 定位 | 参考方法 | 核心论文 |
|------|------|---------|---------|
| **Burst Expert** | 时序突变检测 | MAD-GAN | Li, D., et al. "MAD-GAN: Multivariate Anomaly Detection for Time Series Data using Generative Adversarial Networks." *ICANN*. 2019. |
| **Spectral Expert** | 频域结构异常检测 | GANomaly | Akcay, S., et al. "GANomaly: Semi-Supervised Anomaly Detection via Adversarial Training." *ACCV*. 2018. |
| **Drift Expert** | 统计漂移检测 | Gaussian HMM | 受 HMM 异常检测与对抗数据增强思想启发，参考：Dabrowski, J., et al. "Adversarial Data Augmentation for Hidden Markov Models." *IEEE ICASSP*. 2019. |
| **Consistency Expert** | 传感器一致性断裂检测 | GDN | Deng, A., and Hooi, B. "Graph Neural Network-Based Anomaly Detection in Multivariate Time Series." *AAAI*. 2021. |

> 注：`third_party/` 目录包含上述方法的原始参考代码，仅供本地研究参考，不参与本仓库版本控制。

---

## 当前进展

### Phase 1：数据预处理与基线复现 ✅

- [x] RflyMAD 数据解析与 41 维字段提取
- [x] 滑动窗口（128 帧 / 步长 64）、Z-Score 标准化
- [x] STFT 时频图像生成
- [x] 6 阶段飞行阶段标签
- [x] 数据集划分：train / val / test_closed / test_open
- [x] 分阶段 FP-Growth 等价关联规则挖掘与邻接矩阵生成
- [x] MAD-GAN、GANomaly、HMM、GDN 四大专家复现

### Phase 1.5：物理一致性图 ✅

- [x] 6 个阶段物理一致性邻接矩阵
- [x] GDN 图结构消融（phase / full / identity / global / random）

### Phase 2：核心模块开发（进行中）

- [x] Score-Level Router v1
- [ ] 阶段感知多视图编码器
- [ ] 物理一致性 GNN
- [ ] 稀疏专家路由网络（Top-k）
- [ ] 反事实模态扰动训练

---

## 关键实验结果

### 单专家性能

| Expert | 方法 | 闭集 AUC | 开集 AUC |
|--------|------|---------|---------|
| Burst Expert | MAD-GAN | **0.9168** | **0.8568** |
| Spectral Expert | GANomaly | **0.8544** | **0.7004** |
| Drift Expert | HMM | **0.7819** | **0.7569** |
| Consistency Expert | GDN | **0.9642** | **0.9122** |

### FlightMoE v1：Score-Level Router

| 方法 | Val AUC | Test Closed AUC | Test Open AUC |
|------|---------|-----------------|---------------|
| Single GDN（最佳单专家）| 0.9506 | 0.9491 | 0.8959 |
| Average Fusion | 0.9324 | 0.9207 | 0.8314 |
| Static Weighted | 0.9427 | 0.9261 | 0.8403 |
| Phase-Static Weighted | 0.9459 | 0.9260 | 0.8521 |
| **MLP Router** | **0.9833** | **0.9611** | **0.9138** |

详细结果见 `docs/baseline_results.md` 与 `experiments/router_v1/router_v1_results.json`。

---

## 环境

```bash
conda activate uav_anomaly
```

依赖：PyTorch、PyTorch Geometric、scikit-learn、hmmlearn、numpy、pandas、scipy、matplotlib、tqdm 等。

（TODO：补充 `requirements.txt`）

---

## 快速开始

### 1. 数据预处理

```bash
python src/data/split.py
python src/data/preprocess.py
python src/data/stft_generator.py
python src/data/build_adjacency_matrix.py
```

### 2. 训练基线专家

```bash
# MAD-GAN
python src/baselines/mad_gan/train.py --epochs 50 --batch_size 32

# GANomaly
python src/baselines/ganomaly/train.py --epochs 50 --batch_size 32

# GDN（分阶段训练）
python src/baselines/gdn/train_eval.py

# HMM
python src/baselines/hmm_drift/train_eval.py
```

### 3. 导出专家分数并训练 Router

```bash
python src/baselines/mad_gan/export_scores.py
python src/baselines/ganomaly/export_scores.py
python src/baselines/hmm_drift/export_scores.py
python src/baselines/gdn/train_eval.py --export_scores

python src/models/router_v1.py
```

---

## 后续工作

- [ ] 按故障类型 / 飞行阶段拆解 Router v1 指标
- [ ] 实现阶段感知多视图编码器
- [ ] 实现物理一致性 GNN
- [ ] 实现稀疏专家路由网络（Top-k + 负载均衡损失）
- [ ] 实现反事实模态扰动训练
- [ ] 端到端训练与消融实验
- [ ] AAAI 论文撰写

---

## 许可证

TODO

---

## 致谢

- 数据集：[RflyMAD](https://rfly-openha.github.io/documents/4_resources/dataset.html)
- 数据处理工具：[lerlis/Data_processing_tools](https://github.com/lerlis/Data_processing_tools.git)
- 基线方法：MAD-GAN、GANomaly、GDN 原作者
