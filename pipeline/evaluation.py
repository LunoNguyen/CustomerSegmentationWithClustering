"""evaluation.py

Đánh giá KMeans và KMedoids bằng:
- Silhouette Score: càng cao càng tốt.
- Davies-Bouldin Index: càng thấp càng tốt.
- Calinski-Harabasz Index: càng cao càng tốt.
- Inertia/WCSS cho KMeans để dùng Elbow Method.

Bản tối ưu RAM:
- Silhouette được tính trên mẫu tối đa 5000 dòng vì silhouette_score chính xác
  tạo ma trận khoảng cách cỡ n x n và có thể làm tràn RAM.
"""
from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

from pipeline.clustering import ClusteringOutput, run_kmeans, run_kmedoids


def _cluster_balance_score(labels: np.ndarray) -> float:
    counts = pd.Series(labels).value_counts()
    if counts.empty:
        return 0.0
    return float(counts.min() / counts.max())


def _sample_for_silhouette(
    X: np.ndarray,
    labels: np.ndarray,
    max_samples: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, bool]:
    n_samples = len(X)
    if n_samples <= max_samples:
        return X, labels, False

    rng = np.random.default_rng(random_state)

    # Cố gắng giữ đủ đại diện từng cụm.
    sampled_indices = []
    unique_labels = np.unique(labels)
    per_cluster_min = max(2, max_samples // max(len(unique_labels), 1) // 2)

    for cluster_id in unique_labels:
        cluster_indices = np.where(labels == cluster_id)[0]
        take = min(len(cluster_indices), per_cluster_min)
        if take > 0:
            sampled_indices.extend(rng.choice(cluster_indices, size=take, replace=False).tolist())

    remaining = max_samples - len(sampled_indices)
    if remaining > 0:
        all_indices = np.arange(n_samples)
        already = np.array(sampled_indices, dtype=int)
        mask = np.ones(n_samples, dtype=bool)
        mask[already] = False
        candidates = all_indices[mask]
        take = min(remaining, len(candidates))
        if take > 0:
            sampled_indices.extend(rng.choice(candidates, size=take, replace=False).tolist())

    sampled_indices = np.array(sampled_indices[:max_samples], dtype=int)
    return X[sampled_indices], labels[sampled_indices], True


def _safe_metrics(
    X: np.ndarray,
    labels: np.ndarray,
    random_state: int = 42,
    silhouette_max_samples: int = 5000,
) -> tuple[float, float, float, bool]:
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2 or len(unique_labels) >= len(X):
        return np.nan, np.nan, np.nan, False

    X_sil, labels_sil, sampled = _sample_for_silhouette(
        X=X,
        labels=labels,
        max_samples=silhouette_max_samples,
        random_state=random_state,
    )

    if len(np.unique(labels_sil)) < 2 or len(np.unique(labels_sil)) >= len(X_sil):
        silhouette = np.nan
    else:
        silhouette = silhouette_score(X_sil, labels_sil, metric="euclidean")

    # DBI và CHI không cần ma trận n x n nên vẫn tính trên toàn bộ dữ liệu.
    davies_bouldin = davies_bouldin_score(X, labels)
    calinski_harabasz = calinski_harabasz_score(X, labels)
    return float(silhouette), float(davies_bouldin), float(calinski_harabasz), sampled


def _normalize_for_rank(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    if s.isna().all():
        return pd.Series(np.zeros(len(series)), index=series.index)

    min_value = s.min()
    max_value = s.max()

    if pd.isna(min_value) or pd.isna(max_value) or max_value == min_value:
        normalized = pd.Series(np.ones(len(series)), index=series.index)
    else:
        normalized = (s - min_value) / (max_value - min_value)

    if not higher_is_better:
        normalized = 1 - normalized

    return normalized.fillna(0)


def evaluate_model(
    customer_features_scaled: pd.DataFrame,
    feature_cols: list[str],
    k_values: Iterable[int] = range(2, 11),
    random_state: int = 42,
    silhouette_max_samples: int = 5000,
    kmedoids_max_fit_samples: int = 5000,
) -> Tuple[pd.DataFrame, Dict[tuple[str, int], ClusteringOutput], tuple[str, int]]:
    X = customer_features_scaled[feature_cols].to_numpy(dtype=np.float32)
    n_samples = len(X)

    if n_samples < 3:
        raise ValueError("Cần ít nhất 3 khách hàng để đánh giá phân cụm.")

    max_valid_k = min(max(k_values), n_samples - 1)
    valid_k_values = [k for k in k_values if 2 <= k <= max_valid_k]

    if not valid_k_values:
        raise ValueError("Không có giá trị k hợp lệ. Cần k từ 2 đến số khách hàng - 1.")

    outputs: Dict[tuple[str, int], ClusteringOutput] = {}
    rows = []

    for k in valid_k_values:
        model_outputs = [
            run_kmeans(customer_features_scaled, feature_cols, k, random_state),
            run_kmedoids(
                customer_features_scaled,
                feature_cols,
                k,
                random_state,
                max_fit_samples=kmedoids_max_fit_samples,
            ),
        ]

        for output in model_outputs:
            silhouette, dbi, chi, silhouette_sampled = _safe_metrics(
                X,
                output.labels,
                random_state=random_state,
                silhouette_max_samples=silhouette_max_samples,
            )
            counts = pd.Series(output.labels).value_counts().sort_index()
            outputs[(output.method, k)] = output

            rows.append(
                {
                    "method": output.method,
                    "k": k,
                    "silhouette_score": silhouette,
                    "silhouette_sampled": silhouette_sampled,
                    "davies_bouldin_index": dbi,
                    "calinski_harabasz_index": chi,
                    "inertia_wcss": output.inertia if output.method == "KMeans" else np.nan,
                    "kmedoids_distance": output.inertia if output.method == "KMedoids" else np.nan,
                    "kmedoids_sampled_fit": output.sampled_for_fit if output.method == "KMedoids" else False,
                    "kmedoids_fit_sample_size": output.fit_sample_size if output.method == "KMedoids" else np.nan,
                    "cluster_balance_score": _cluster_balance_score(output.labels),
                    "min_cluster_size": int(counts.min()),
                    "max_cluster_size": int(counts.max()),
                }
            )

    evaluation_df = pd.DataFrame(rows)

    evaluation_df["rank_score"] = (
        0.40 * _normalize_for_rank(evaluation_df["silhouette_score"], True)
        + 0.25 * _normalize_for_rank(evaluation_df["davies_bouldin_index"], False)
        + 0.20 * _normalize_for_rank(evaluation_df["calinski_harabasz_index"], True)
        + 0.15 * _normalize_for_rank(evaluation_df["cluster_balance_score"], True)
    )

    evaluation_df = evaluation_df.sort_values(
        by=["rank_score", "silhouette_score", "davies_bouldin_index"],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    best_method = str(evaluation_df.loc[0, "method"])
    best_k = int(evaluation_df.loc[0, "k"])

    return evaluation_df, outputs, (best_method, best_k)
