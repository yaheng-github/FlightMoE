"""
HMM Drift Expert 简化基线
核心：滑动窗口 + GaussianHMM + Hellinger 距离
检测缓慢漂移和长期失配
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from hmmlearn import hmm
from scipy import linalg, stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, precision_recall_curve


def hellinger_dist(mu_model, cov_model_in, mu_data, cov_data_in):
    """Hellinger 距离：衡量模型分布与数据分布的差异"""
    # 兼容 hmmlearn 新旧版本：输入可能是向量（diag）或矩阵（full/diag 矩阵）
    if cov_model_in.ndim == 1:
        cov_model = np.diag(cov_model_in)
    else:
        cov_model = cov_model_in
    if cov_data_in.ndim == 1:
        cov_data = np.diag(cov_data_in)
    else:
        cov_data = cov_data_in
    num_comp1 = (linalg.det(cov_model) ** (1 / 4)) * (linalg.det(cov_data) ** (1 / 4))
    den_comp1 = (linalg.det((cov_model + cov_data) / 2) ** (1 / 2))
    comp1 = num_comp1 / den_comp1
    comp2 = float(np.exp((-1 / 8) * (mu_model - mu_data) @ np.linalg.matrix_power((cov_model + cov_data) / 2, -1) @ (mu_model - mu_data).T))
    return 1 - comp1 * comp2


def bic(size, n_states, n_comp, ll):
    p = n_states ** 2 + n_comp * n_states - 1
    return -2 * ll + p * np.log(size)


def train_hmm(train_npz, pca_components=4, max_states=15, window_size=100):
    """训练 HMM + 阈值"""
    data = np.load(train_npz)
    temporal = data['temporal'].astype(np.float32)  # [N, 128, 41]
    anomaly_labels = data['anomaly_labels']

    # 只使用正常样本训练
    normal_data = temporal[anomaly_labels == 0]
    # 展平每个窗口为向量 [N, 128*41]
    train_vec = normal_data.reshape(normal_data.shape[0], -1)

    # PCA 降维
    if pca_components > 0:
        sc = StandardScaler()
        pca = PCA(n_components=pca_components)
        train_vec = sc.fit_transform(train_vec)
        train_vec = pca.fit_transform(train_vec)
    else:
        sc, pca = None, None

    # BIC 选最优状态数
    print("[HMM] BIC 选择最优状态数...")
    bics = []
    for K in range(2, max_states):
        try:
            np.random.seed(0)
            model = hmm.GaussianHMM(n_components=K, covariance_type="diag", n_iter=100, random_state=0).fit(train_vec)
            ll = model.score(train_vec)
            bics.append(bic(train_vec.shape[0], K, 2 * train_vec.shape[1], ll))
        except Exception:
            bics.append(np.inf)

    K = np.argmin(bics) + 2
    print(f"[HMM] 最优状态数: {K}")

    # 训练最终模型
    np.random.seed(0)
    model = hmm.GaussianHMM(n_components=K, covariance_type="diag", n_iter=100, random_state=0)
    model.fit(train_vec)

    # 阈值训练：在正常数据上滑动窗口计算最大 Hellinger 距离
    print("[HMM] 训练阈值...")
    scores = []
    for i in range(window_size, train_vec.shape[0]):
        Wt = train_vec[i - window_size:i].copy()
        ll, St = model.decode(Wt)
        st = int(stats.mode(St, keepdims=True).mode[0])
        X = Wt[St == st]
        mu = np.reshape(np.mean(X, axis=0), [1, train_vec.shape[1]])
        cov = (np.diag(np.cov(X.T)) + 1e-5) * np.eye(train_vec.shape[1], train_vec.shape[1])
        score = hellinger_dist(model.means_[st], model.covars_[st], mu, cov)
        scores.append(score)

    thresh = np.max(scores) + 0.01  # 加一点余量
    print(f"[HMM] 阈值: {thresh:.4f}")

    return model, thresh, sc, pca, window_size


def evaluate_hmm(model, thresh, sc, pca, window_size, test_npz):
    """评估"""
    data = np.load(test_npz)
    temporal = data['temporal'].astype(np.float32)
    labels = data['anomaly_labels']

    # 展平 + 降维
    test_vec = temporal.reshape(temporal.shape[0], -1)
    if sc is not None and pca is not None:
        test_vec = sc.transform(test_vec)
        test_vec = pca.transform(test_vec)

    scores = []
    for i in range(window_size, test_vec.shape[0]):
        Wt = test_vec[i - window_size:i].copy()
        ll, St = model.decode(Wt)
        st = stats.mode(St)[0]
        X = Wt[St == st]
        mu = np.reshape(np.mean(X, axis=0), [1, test_vec.shape[1]])
        cov = (np.diag(np.cov(X.T)) + 1e-5) * np.eye(test_vec.shape[1], test_vec.shape[1])
        score = hellinger_dist(model.means_[st], model.covars_[st], mu, cov)
        scores.append(score)

    scores = np.array(scores)
    labels = labels[window_size:]
    labels = labels[:len(scores)]

    auc = roc_auc_score(labels, scores)

    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-10)
    best_idx = np.argmax(f1_scores)
    best_th = thresholds[best_idx] if best_idx < len(thresholds) else thresholds[-1]
    preds = (scores >= best_th).astype(int)
    f1 = f1_score(labels, preds)
    precision_val = precision_score(labels, preds)
    recall_val = recall_score(labels, preds)

    return auc, f1, precision_val, recall_val, thresh


def main():
    import argparse
    parser = argparse.ArgumentParser(description="HMM Drift Expert")
    parser.add_argument("--train_npz", type=str, default="./data/preprocessed/train.npz")
    parser.add_argument("--test_closed", type=str, default="./data/preprocessed/test_closed.npz")
    parser.add_argument("--test_open", type=str, default="./data/preprocessed/test_open.npz")
    parser.add_argument("--pca", type=int, default=4)
    parser.add_argument("--max_states", type=int, default=15)
    parser.add_argument("--window_size", type=int, default=100)
    parser.add_argument("--output_dir", type=str, default="./checkpoints/hmm_drift")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("HMM Drift Expert 基线")
    print("=" * 70)

    print("\n[Step 1/2] 训练 HMM...")
    model, thresh, sc, pca, w = train_hmm(
        args.train_npz,
        pca_components=args.pca,
        max_states=args.max_states,
        window_size=args.window_size,
    )

    # 保存模型
    import pickle
    with open(f"{args.output_dir}/hmm_model.pkl", "wb") as f:
        pickle.dump({"model": model, "thresh": thresh, "sc": sc, "pca": pca, "w": w}, f)

    print("\n[Step 2/2] 评估...")
    for name, path in [("Closed", args.test_closed), ("Open", args.test_open)]:
        auc, f1, prec, rec, th = evaluate_hmm(model, thresh, sc, pca, w, path)
        print(f"\n{name} Set:")
        print(f"  AUC-ROC : {auc:.4f}")
        print(f"  F1      : {f1:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall   : {rec:.4f}")
        print(f"  Threshold: {th:.4f}")

    print("\n[DONE]")


if __name__ == "__main__":
    main()
