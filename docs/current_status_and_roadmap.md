# FlightMoE Current Status and Roadmap

Updated: 2026-05-29

## 1. Current Project State

FlightMoE has completed the data pipeline and the four baseline experts needed for a score-level MoE prototype.

### Data Pipeline

- Raw RflyMAD cases are organized under `Data_processing_tools/RflyMAD dataset/`.
- `src/data/split.py` builds `train`, `val`, `test_closed`, and `test_open` splits by CaseID.
- `src/data/preprocess.py` extracts 41 UAV sensor features and sliding windows of shape `[128, 41]`.
- `src/data/stft_generator.py` builds spectral features for the frequency-domain expert.
- `src/data/build_adjacency_matrix.py` builds six phase-specific adjacency matrices.

Prepared window counts:

| Split | Windows | Cases | Normal | Anomaly | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| train | 5,300 | 365 | 5,300 | 0 | Normal-only training split |
| val | 17,263 | 828 | 8,574 | 8,689 | Threshold/router validation |
| test_closed | 69,134 | 3,447 | 32,048 | 37,086 | Known fault types |
| test_open | 27,946 | 1,439 | 14,076 | 13,870 | Unseen fault types plus real normal domain shift |

Current open-set wording should be precise: `test_open` evaluates unseen fault-type generalization. Real-flight samples currently appear as normal samples, so they support real-normal domain-shift analysis, not real anomalous-flight evaluation.

### Expert Pool

| Expert | Role | Main path | Closed AUC | Open AUC | Status |
| --- | --- | --- | ---: | ---: | --- |
| MAD-GAN | Burst Expert | `src/baselines/mad_gan/` | 0.9168 | 0.8568 | Complete |
| GANomaly | Spectral Expert | `src/baselines/ganomaly/` | 0.8544 | 0.7004 | Complete |
| HMM | Drift Expert | `src/baselines/hmm_drift/` | 0.7819 | 0.7569 | Complete |
| GDN | Consistency Expert | `src/baselines/gdn/` | 0.9642 | 0.9122 | Complete |

The best single expert is currently GDN, which supports the value of sensor-consistency modeling.

## 2. Physical Consistency Graph Positioning

The current adjacency matrices should be described as phase-aware, physics-inspired statistical priors mined from normal data.

They should not be described as hard physical constraints or first-principles dynamics. The graph is relatively dense before GDN applies its internal `topk=5` neighbor selection, so the clearest description is:

> phase-specific soft candidate graphs plus sparse Top-K message passing inside the Consistency Expert.

Recommended graph ablations:

- phase-specific mined graph
- global mined graph
- full graph
- identity graph
- random graph

The repository already contains GDN graph-ablation score exports under `experiments/scores/`.

## 3. Immediate Missing Work

The next project milestone is FlightMoE v1: score-level expert fusion.

Required inputs:

- `experiments/scores/madgan_{split}.npz`
- `experiments/scores/ganomaly_{split}.npz`
- `experiments/scores/hmm_{split}.npz`
- `experiments/scores/gdn_phase_{split}.npz`

Current status:

- HMM score files exist.
- GDN score files for multiple graph types exist.
- MAD-GAN and GANomaly score exports still need to be generated in `experiments/scores/`.
- `src/models/router_v1.py` already implements a score-level Router MLP and comparison baselines.

Recommended execution order:

```bash
python src/baselines/mad_gan/export_scores.py
python src/baselines/ganomaly/export_scores.py
python src/models/router_v1.py
```

The MAD-GAN score export can take noticeable time because it optimizes latent variables for each batch.

## 4. FlightMoE v1 Experiment Plan

Use frozen expert scores and train only the fusion/router layer on `val`.

Report:

- single experts
- average fusion
- static weighted fusion
- phase-static weighted fusion
- MLP Router

Key checks:

- Router should beat average fusion.
- Router should not collapse to only one expert for all samples.
- Open-set performance should be competitive with GDN; if AUC does not exceed GDN, analyze whether F1, precision, fault-type robustness, or interpretability improves.

Required analysis tables:

- metrics by split
- metrics by fault type
- metrics by flight phase
- router mean weights by phase
- router weights by fault type

## 5. Later Work

After FlightMoE v1 is stable:

- add Top-2 sparse routing and compare against dense routing
- add reliability features such as missing masks, sensor-group noise estimates, and uncertainty proxies
- implement counterfactual perturbations for missing, jitter, noise, spectral texture, domain shift, and physics-breaking cases
- build a stronger multi-view encoder only after score-level fusion has established the experimental story

