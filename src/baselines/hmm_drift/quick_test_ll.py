"""快速验证：用 log-likelihood 代替 Hellinger 距离"""
import numpy as np
from hmmlearn import hmm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

# 加载数据
print("[1/2] 加载数据并降维...")
train = np.load("./data/preprocessed/train.npz")
test = np.load("./data/preprocessed/test_closed.npz")

train_seq = train['temporal'].reshape(-1, train['temporal'].shape[-1])
test_seq = test['temporal'].reshape(-1, test['temporal'].shape[-1])
test_labels = np.repeat(test['anomaly_labels'], test['temporal'].shape[1])

sc = StandardScaler()
pca = PCA(n_components=4)
train_seq = sc.fit_transform(train_seq)
train_seq = pca.fit_transform(train_seq)
test_seq = sc.transform(test_seq)
test_seq = pca.transform(test_seq)

# 降采样
STRIDE = 20
train_seq = train_seq[::STRIDE]
test_seq = test_seq[::STRIDE]
test_labels = test_labels[::STRIDE]

# 训练 HMM
print("[2/2] 训练 HMM (K=7) 并计算 log-likelihood...")
model = hmm.GaussianHMM(n_components=7, covariance_type="diag", n_iter=100, random_state=0)
model.fit(train_seq)

# 滑动窗口计算两种分数
WINDOW = 100
np.random.seed(0)
normal_idx = np.where(test_labels == 0)[0]
anomaly_idx = np.where(test_labels == 1)[0]
sample_normal = np.random.choice(normal_idx[normal_idx >= WINDOW], 500, replace=False)
sample_anomaly = np.random.choice(anomaly_idx[anomaly_idx >= WINDOW], 500, replace=False)

def calc_ll(indices):
    scores = []
    for i in indices:
        Wt = test_seq[i - WINDOW:i]
        ll = model.score(Wt)
        scores.append(ll)
    return np.array(scores)

ll_normal = calc_ll(sample_normal)
ll_anomaly = calc_ll(sample_anomaly)

print(f"\n正常样本 log-likelihood: mean={ll_normal.mean():.2f}, std={ll_normal.std():.2f}")
print(f"异常样本 log-likelihood: mean={ll_anomaly.mean():.2f}, std={ll_anomaly.std():.2f}")

# 计算 AUC（注意：log-likelihood 越高越正常，所以异常分数 = -log-likelihood）
all_scores = np.concatenate([ll_normal, ll_anomaly])
all_labels = np.array([0]*500 + [1]*500)
auc = roc_auc_score(all_labels, -all_scores)
print(f"\n用 -log-likelihood 作为异常分数的 AUC: {auc:.4f}")
