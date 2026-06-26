"""快速验证：窗口级统计特征 + HMM"""
import numpy as np
from hmmlearn import hmm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, precision_recall_curve

def extract_features(temporal):
    """提取窗口级统计特征 [N, 128, 41] -> [N, D]"""
    # 均值
    mean_feat = temporal.mean(axis=1)  # [N, 41]
    # 标准差
    std_feat = temporal.std(axis=1)    # [N, 41]
    # 最大值
    max_feat = temporal.max(axis=1)    # [N, 41]
    # 最小值
    min_feat = temporal.min(axis=1)    # [N, 41]
    # 极差
    range_feat = max_feat - min_feat   # [N, 41]
    # 拼接
    features = np.concatenate([mean_feat, std_feat, range_feat], axis=1)  # [N, 123]
    return features

print("[1/3] 加载数据并提取统计特征...")
train = np.load("./data/preprocessed/train.npz")
test = np.load("./data/preprocessed/test_closed.npz")

train_feat = extract_features(train['temporal'])
test_feat = extract_features(test['temporal'])
test_labels = test['anomaly_labels']

print(f"  训练特征: {train_feat.shape}, 测试特征: {test_feat.shape}")

# 标准化 + PCA
sc = StandardScaler()
pca = PCA(n_components=8)
train_feat = sc.fit_transform(train_feat)
train_feat = pca.fit_transform(train_feat)
test_feat = sc.transform(test_feat)
test_feat = pca.transform(test_feat)

# 训练 HMM
print("[2/3] 训练 HMM...")
model = hmm.GaussianHMM(n_components=7, covariance_type="diag", n_iter=100, random_state=0)
model.fit(train_feat)

# 计算 log-likelihood 作为异常分数
print("[3/3] 评估...")
scores = []
for i in range(len(test_feat)):
    # 用单个样本的 score（HMM 支持单个样本）
    ll = model.score(test_feat[i:i+1])
    scores.append(ll)

scores = -np.array(scores)  # 取负：越高越异常

auc = roc_auc_score(test_labels, scores)
precision, recall, thresholds = precision_recall_curve(test_labels, scores)
f1_scores = 2 * precision * recall / (precision + recall + 1e-10)
best_idx = np.argmax(f1_scores)
best_th = thresholds[best_idx] if best_idx < len(thresholds) else thresholds[-1]
preds = (scores >= best_th).astype(int)
f1 = f1_score(test_labels, preds)
prec = precision_score(test_labels, preds)
rec = recall_score(test_labels, preds)

print(f"\n闭集测试 (窗口级统计特征 + HMM log-likelihood):")
print(f"  AUC-ROC : {auc:.4f}")
print(f"  F1      : {f1:.4f}")
print(f"  Precision: {prec:.4f}")
print(f"  Recall   : {rec:.4f}")
