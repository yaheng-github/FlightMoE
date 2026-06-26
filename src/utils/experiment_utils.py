"""Shared utilities for FlightMoE v1 experiments."""

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import (
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


FAULT_TYPE_ID_MAP = {
    0: "motor",
    1: "propeller",
    2: "low_voltage",
    3: "wind_affect",
    4: "load_lose",
    5: "accelerometer",
    6: "gyroscope",
    7: "magnetometer",
    8: "barometer",
    9: "GPS",
    10: "no_fault",
}

DATA_TYPE_ID_MAP = {1: "SIL", 2: "HIL", 3: "Real"}


def parse_case_id(case_id: int) -> Tuple[int, int, int, int]:
    data_type = case_id // 1000000000
    flight_mode = (case_id % 1000000000) // 100000000
    fault_type = (case_id % 100000000) // 1000000
    case_num = case_id % 1000000
    return data_type, flight_mode, fault_type, case_num


def load_npz_metadata(npz_path: str) -> Dict[str, np.ndarray]:
    data = np.load(npz_path)
    case_ids = data["case_ids"].astype(np.int64)
    parsed = np.array([parse_case_id(int(cid)) for cid in case_ids], dtype=np.int64)
    return {
        "labels": data["anomaly_labels"].astype(np.int32),
        "phase_labels": data["phase_labels"].astype(np.int32),
        "case_ids": case_ids,
        "data_types": parsed[:, 0].astype(np.int32),
        "fault_types": parsed[:, 2].astype(np.int32),
    }


def compute_metrics(labels: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    labels = np.asarray(labels).astype(np.int32)
    scores = np.asarray(scores).astype(np.float64)
    if len(labels) != len(scores):
        raise ValueError(f"labels/scores length mismatch: {len(labels)} vs {len(scores)}")
    if len(np.unique(labels)) < 2:
        raise ValueError("metrics require both normal and anomaly labels")

    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-10)
    best_idx = int(np.argmax(f1_scores))
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else thresholds[-1]
    preds = (scores >= best_threshold).astype(np.int32)

    return {
        "auc_roc": float(roc_auc_score(labels, scores)),
        "f1": float(f1_score(labels, preds)),
        "precision": float(precision_score(labels, preds)),
        "recall": float(recall_score(labels, preds)),
        "best_threshold": float(best_threshold),
        "n_samples": int(len(labels)),
        "n_anomalies": int(labels.sum()),
    }


def save_json(path: str, payload: Dict):
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def save_score_npz(path: str, npz_path: str, expert: str, split: str, raw_scores: np.ndarray, **extra):
    meta = load_npz_metadata(npz_path)
    raw_scores = np.asarray(raw_scores, dtype=np.float32)
    if len(raw_scores) != len(meta["labels"]):
        raise ValueError(f"{expert} score length mismatch for {split}: {len(raw_scores)} vs {len(meta['labels'])}")

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        expert=np.array(expert),
        split=np.array(split),
        source_npz=np.array(npz_path),
        score_raw=raw_scores,
        label=meta["labels"],
        phase_label=meta["phase_labels"],
        case_id=meta["case_ids"],
        fault_type=meta["fault_types"],
        data_type=meta["data_types"],
        **extra,
    )


def robust_scale_from_train(values: np.ndarray, ref_values: np.ndarray) -> np.ndarray:
    ref_values = np.asarray(ref_values, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    median = np.median(ref_values)
    q1, q3 = np.percentile(ref_values, [25, 75])
    iqr = max(q3 - q1, 1e-8)
    scaled = (values - median) / iqr
    return scaled.astype(np.float32)
