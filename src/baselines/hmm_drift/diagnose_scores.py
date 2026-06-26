"""诊断 HMM Hellinger 分数分布"""
import numpy as np
from hmmlearn import hmm
from scipy import linalg, stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def hellinger_dist(mu_model, cov_model, mu_data, cov_data):
    if cov_model.ndim == 1:
        cov_model = np.diag(cov_model)
    if cov_data.ndim == 1:
        cov_data = np.diag(cov_data)
    num_comp1 = (linalg.det(cov_model) ** (1 / 4)) * (linalg.det(cov_data) ** (1 / 4))
    den_comp1 = (linalg.det((cov_model + cov_data) / 2) ** (1 / 2))
    comp1 = num_comp1 / den_comp1
    comp2 = float(np.exp((-1 / 8) * (mu_model - mu_data) @ np.linalg.matrix_power((cov_model + cov_data) / 2, -1) @ (mu_model - mu_data).T))
    return 1 - comp1 * comp2

# 加载数据
print("[1/3] 加载数据...")
train = np.load("./data/preprocessed/train.npz")
test = np.load("./data/preprocessed/test_closed.npz")

train_seq = train['temporal'].reshape(-1, train['temporal'].shape[-1])
test_seq = test['temporal'].reshape(-1, test['temporal'].shape[-1])
test_labels = np.repeat(test['anomaly_labels'], test['temporal'].shape[1])

# PCA
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
print("[2/3] 训练 HMM (K=7)...")
model = hmm.GaussianHMM(n_components=7, covariance_type="diag", n_iter=100, random_state=0)
model.fit(train_seq)

# 采样计算 scores
print("[3/3] 采样计算 Hellinger 距离...")
WINDOW = 100
np.random.seed(0)
normal_idx = np.where(test_labels == 0)[0]
anomaly_idx = np.where(test_labels == 1)[0]

# 各随机采样 500 个窗口位置
sample_normal = np.random.choice(normal_idx[normal_idx >= WINDOW], 500, replace=False)
sample_anomaly = np.random.choice(anomaly_idx[anomaly_idx >= WINDOW], 500, replace=False)

def calc_scores(indices):
    scores = []
    for i in indices:
        Wt = test_seq[i - WINDOW:i].copy()
        _, St = model.decode(Wt)
        st = int(stats.mode(St, keepdims=True).mode[0])
        X = Wt[St == st]
        if len(X) < 2:
            scores.append(0)
            continue
        mu = np.reshape(np.mean(X, axis=0), [1, test_seq.shape[1]])
        cov = (np.diag(np.cov(X.T)) + 1e-5) * np.eye(test_seq.shape[1], test_seq.shape[1])
        score = hellinger_dist(model.means_[st], model.covars_[st], mu, cov)
        scores.append(score)
    return np.array(scores)

scores_normal = calc_scores(sample_normal)
scores_anomaly = calc_scores(sample_anomaly)

print(f"\n正常样本 Hellinger 距离: mean={scores_normal.mean():.4f}, std={scores_normal.std():.4f}, min={scores_normal.min():.4f}, max={scores_normal.max():.4f}")
print(f"异常样本 Hellinger 距离: mean={scores_anomaly.mean():.4f}, std={scores_anomaly.std():.4f}, min={scores_anomaly.min():.4f}, max={scores_anomaly.max():.4f}")
print(f"\n正常样本 score > 0.9 比例: {(scores_normal > 0.9).mean():.2%}")
print(f"异常样本 score > 0.9 比例: {(scores_anomaly > 0.9).mean():.2%}")
print(f"\n标签分布: 正常={(test_labels==0).sum()}, 异常={(test_labels==1).sum()}, 异常比例={(test_labels==1).mean():.2%}")
