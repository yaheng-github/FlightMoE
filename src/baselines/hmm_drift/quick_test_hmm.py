"""快速验证 HMM 在降采样数据上能否跑通，并计时"""
import time
import numpy as np
from hmmlearn import hmm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

DATA = "./data/preprocessed/train.npz"
STRIDE = 20
MAX_STATES = 8

print("[1/4] 加载数据...")
t0 = time.time()
data = np.load(DATA)
seq = data['temporal'].reshape(-1, data['temporal'].shape[-1])
print(f"  原始序列: {seq.shape}, 耗时 {time.time()-t0:.2f}s")

print("[2/4] PCA 降维...")
t0 = time.time()
sc = StandardScaler()
pca = PCA(n_components=4)
seq = sc.fit_transform(seq)
seq = pca.fit_transform(seq)
print(f"  PCA 后: {seq.shape}, 耗时 {time.time()-t0:.2f}s")

print(f"[3/4] 降采样 stride={STRIDE}...")
seq = seq[::STRIDE]
print(f"  降采样后: {seq.shape}")

print(f"[4/4] BIC 搜索状态数 2~{MAX_STATES}...")
for K in range(2, MAX_STATES):
    t0 = time.time()
    model = hmm.GaussianHMM(n_components=K, covariance_type="diag", n_iter=100, random_state=0)
    model.fit(seq)
    ll = model.score(seq)
    elapsed = time.time() - t0
    print(f"  K={K}: log-likelihood={ll:.2f}, 耗时 {elapsed:.1f}s")

print("\n[OK] HMM 训练可以跑通！")
