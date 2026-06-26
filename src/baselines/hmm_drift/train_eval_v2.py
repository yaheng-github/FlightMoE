"""
HMM Drift Expert —— 适配 RflyMAD 数据版本
核心改进：窗口级统计特征 + HMM log-likelihood
原因：原始时间步级 HMM 在窗口化数据上失效（窗口边界不连续、41维传感器复杂）
      窗口级统计特征（均值/标准差/极差）能稳定刻画窗口分布，HMM 学习正常窗口模式
"""
import argparse
import pickle
from pathlib import Path

import numpy as np
from hmmlearn import hmm
from sklearn.decomposition import PCA
from sklearn.metrics import (f1_score, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score)
from sklearn.preprocessing import StandardScaler


def extract_features(temporal):
    """提取窗口级统计特征 [N, 128, 41] -> [N, D]"""
    mean_feat = temporal.mean(axis=1)          # [N, 41]
    std_feat = temporal.std(axis=1)            # [N, 41]
    range_feat = temporal.max(axis=1) - temporal.min(axis=1)  # [N, 41]
    return np.concatenate([mean_feat, std_feat, range_feat], axis=1)  # [N, 123]


def bic(size, n_states, n_comp, ll):
    p = n_states ** 2 + n_comp * n_states - 1
    return -2 * ll + p * np.log(size)


def train_and_eval(train_npz, test_npz, pca_components=8, max_states=15):
    """训练 + 评估"""
    # -------- 加载数据并提取统计特征 --------
    train_data = np.load(train_npz)
    test_data = np.load(test_npz)

    train_feat = extract_features(train_data['temporal'])
    test_feat = extract_features(test_data['temporal'])
    test_labels = test_data['anomaly_labels']

    # 只使用正常样本训练
    train_labels = train_data['anomaly_labels']
    train_feat = train_feat[train_labels == 0]

    print(f"[HMM] 训练特征: {train_feat.shape}, 测试特征: {test_feat.shape}")

    # -------- PCA 降维 --------
    sc = StandardScaler()
    pca = PCA(n_components=pca_components)
    train_feat = sc.fit_transform(train_feat)
    train_feat = pca.fit_transform(train_feat)
    test_feat = sc.transform(test_feat)
    test_feat = pca.transform(test_feat)

    # -------- BIC 选最优状态数 --------
    print("[HMM] BIC 选择最优状态数...")
    bics = []
    for K in range(2, max_states):
        try:
            np.random.seed(0)
            model = hmm.GaussianHMM(n_components=K, covariance_type="diag", n_iter=100, random_state=0)
            model.fit(train_feat)
            ll = model.score(train_feat)
            bics.append(bic(train_feat.shape[0], K, 2 * train_feat.shape[1], ll))
        except Exception:
            bics.append(np.inf)

    K = np.argmin(bics) + 2
    print(f"[HMM] 最优状态数: {K}")

    # -------- 训练最终 HMM --------
    np.random.seed(0)
    model = hmm.GaussianHMM(n_components=K, covariance_type="diag", n_iter=100, random_state=0)
    model.fit(train_feat)

    # -------- 阈值训练 --------
    print("[HMM] 训练阈值...")
    scores = []
    for i in range(len(train_feat)):
        ll = model.score(train_feat[i:i + 1])
        scores.append(-ll)  # 异常分数：越高越异常
    thresh = np.max(scores) + 0.01
    print(f"[HMM] 阈值: {thresh:.4f}")

    # -------- 测试评估 --------
    print("[HMM] 测试评估...")
    scores = []
    for i in range(len(test_feat)):
        ll = model.score(test_feat[i:i + 1])
        scores.append(-ll)

    scores = np.array(scores)
    labels = test_labels

    auc = roc_auc_score(labels, scores)

    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-10)
    best_idx = np.argmax(f1_scores)
    best_th = thresholds[best_idx] if best_idx < len(thresholds) else thresholds[-1]
    preds = (scores >= best_th).astype(int)
    f1 = f1_score(labels, preds)
    prec = precision_score(labels, preds)
    rec = recall_score(labels, preds)

    return auc, f1, prec, rec, thresh, best_th, model, sc, pca


def main():
    parser = argparse.ArgumentParser(description="HMM Drift Expert")
    parser.add_argument("--train_npz", type=str, default="./data/preprocessed/train.npz")
    parser.add_argument("--test_closed", type=str, default="./data/preprocessed/test_closed.npz")
    parser.add_argument("--test_open", type=str, default="./data/preprocessed/test_open.npz")
    parser.add_argument("--pca", type=int, default=8)
    parser.add_argument("--max_states", type=int, default=15)
    parser.add_argument("--output_dir", type=str, default="./checkpoints/hmm_drift")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("HMM Drift Expert 基线（窗口级统计特征 + HMM log-likelihood）")
    print("=" * 70)

    for name, path in [("Closed", args.test_closed), ("Open", args.test_open)]:
        print(f"\n[{name} Set]")
        auc, f1, prec, rec, thresh, best_th, model, sc, pca = train_and_eval(
            args.train_npz, path,
            pca_components=args.pca,
            max_states=args.max_states,
        )
        print(f"  AUC-ROC : {auc:.4f}")
        print(f"  F1      : {f1:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall   : {rec:.4f}")
        print(f"  Threshold: {thresh:.4f}")
        print(f"  Best F1 Th: {best_th:.4f}")

    # 保存模型
    with open(f"{args.output_dir}/hmm_model_v2.pkl", "wb") as f:
        pickle.dump({"model": model, "sc": sc, "pca": pca}, f)

    print("\n[DONE]")


if __name__ == "__main__":
    main()
