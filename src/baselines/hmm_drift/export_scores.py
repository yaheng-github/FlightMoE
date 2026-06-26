"""Export window-aligned HMM Drift Expert scores for FlightMoE v1."""

import argparse
import pickle
from pathlib import Path
import sys

import numpy as np
from hmmlearn import hmm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parents[2]))

from utils.experiment_utils import save_score_npz


def extract_features(temporal):
    mean_feat = temporal.mean(axis=1)
    std_feat = temporal.std(axis=1)
    range_feat = temporal.max(axis=1) - temporal.min(axis=1)
    return np.concatenate([mean_feat, std_feat, range_feat], axis=1)


def bic(size, n_states, n_comp, ll):
    p = n_states ** 2 + n_comp * n_states - 1
    return -2 * ll + p * np.log(size)


def train_hmm(train_npz, pca_components=8, max_states=15, seed=0):
    train_data = np.load(train_npz)
    train_feat = extract_features(train_data["temporal"])
    train_feat = train_feat[train_data["anomaly_labels"] == 0]
    scaler = StandardScaler()
    pca = PCA(n_components=pca_components)
    train_feat = pca.fit_transform(scaler.fit_transform(train_feat))

    bics = []
    for k in range(2, max_states):
        try:
            np.random.seed(seed)
            model = hmm.GaussianHMM(n_components=k, covariance_type="diag", n_iter=100, random_state=seed)
            model.fit(train_feat)
            bics.append(bic(train_feat.shape[0], k, 2 * train_feat.shape[1], model.score(train_feat)))
        except Exception:
            bics.append(np.inf)
    n_components = int(np.argmin(bics) + 2)
    model = hmm.GaussianHMM(n_components=n_components, covariance_type="diag", n_iter=100, random_state=seed)
    model.fit(train_feat)
    return model, scaler, pca


def score_split(model, scaler, pca, npz_path):
    data = np.load(npz_path)
    feat = pca.transform(scaler.transform(extract_features(data["temporal"])))
    scores = np.array([-model.score(feat[i:i + 1]) for i in range(len(feat))], dtype=np.float32)
    return scores


def main():
    parser = argparse.ArgumentParser(description="Export HMM drift scores")
    parser.add_argument("--train_npz", type=str, default="./data/preprocessed/train.npz")
    parser.add_argument("--val_npz", type=str, default="./data/preprocessed/val.npz")
    parser.add_argument("--test_closed", type=str, default="./data/preprocessed/test_closed.npz")
    parser.add_argument("--test_open", type=str, default="./data/preprocessed/test_open.npz")
    parser.add_argument("--output_dir", type=str, default="./experiments/scores")
    parser.add_argument("--model_out", type=str, default="./checkpoints/hmm_drift/hmm_model_flightmoe_v1.pkl")
    parser.add_argument("--pca", type=int, default=8)
    parser.add_argument("--max_states", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    model, scaler, pca = train_hmm(args.train_npz, pca_components=args.pca, max_states=args.max_states, seed=args.seed)
    with open(args.model_out, "wb") as f:
        pickle.dump({"model": model, "scaler": scaler, "pca": pca}, f)

    for split_name, npz_path in [("val", args.val_npz), ("test_closed", args.test_closed), ("test_open", args.test_open)]:
        scores = score_split(model, scaler, pca, npz_path)
        out_path = Path(args.output_dir) / f"hmm_{split_name}.npz"
        save_score_npz(str(out_path), npz_path, "hmm", split_name, scores)
        print(f"[SAVED] {out_path} ({len(scores)} scores)")


if __name__ == "__main__":
    main()
