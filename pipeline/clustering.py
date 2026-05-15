"""
clustering.py

Gom cụm mall bằng:
- KMeans
- KMedoids nếu sklearn_extra có sẵn, nếu không sẽ dùng fallback đơn giản.
- Hierarchical / AgglomerativeClustering

Dữ liệu gom cụm là bảng mall_features_scaled, mỗi mall là một điểm.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import pairwise_distances


@dataclass
class ClusteringOutput:
    method: str
    labels: np.ndarray
    centers: np.ndarray
    feature_cols: List[str]


def _compute_centers_from_labels(X: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    """Tính tâm cụm bằng trung bình các điểm thuộc cụm."""
    centers = []
    for cluster_id in range(k):
        cluster_points = X[labels == cluster_id]
        if len(cluster_points) == 0:
            centers.append(np.zeros(X.shape[1]))
        else:
            centers.append(cluster_points.mean(axis=0))
    return np.array(centers)


def _kmedoids_fallback(
    X: np.ndarray,
    k: int,
    random_state: int = 42,
    max_iter: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """
    KMedoids fallback đơn giản khi chưa cài scikit-learn-extra.

    Ý tưởng:
    - Khởi tạo medoids ngẫu nhiên.
    - Gán điểm vào medoid gần nhất.
    - Với mỗi cụm, chọn điểm có tổng khoảng cách nhỏ nhất làm medoid mới.
    """
    rng = np.random.default_rng(random_state)

    if len(X) < k:
        raise ValueError("Số điểm nhỏ hơn số cụm k.")

    medoid_indices = rng.choice(len(X), size=k, replace=False)

    for _ in range(max_iter):
        distances = pairwise_distances(X, X[medoid_indices], metric="euclidean")
        labels = distances.argmin(axis=1)

        new_medoid_indices = medoid_indices.copy()

        for cluster_id in range(k):
            cluster_indices = np.where(labels == cluster_id)[0]
            if len(cluster_indices) == 0:
                continue

            cluster_distances = pairwise_distances(
                X[cluster_indices],
                X[cluster_indices],
                metric="euclidean",
            )
            total_distances = cluster_distances.sum(axis=1)
            best_local_index = total_distances.argmin()
            new_medoid_indices[cluster_id] = cluster_indices[best_local_index]

        if np.array_equal(medoid_indices, new_medoid_indices):
            break

        medoid_indices = new_medoid_indices

    distances = pairwise_distances(X, X[medoid_indices], metric="euclidean")
    labels = distances.argmin(axis=1)
    centers = X[medoid_indices]

    return labels, centers


def run_clustering_models(
    mall_features_scaled: pd.DataFrame,
    feature_cols: List[str],
    k: int = 3,
    random_state: int = 42,
) -> Dict[str, ClusteringOutput]:
    """Chạy 3 phương pháp gom cụm."""
    X = mall_features_scaled[feature_cols].to_numpy(dtype=float)

    outputs: Dict[str, ClusteringOutput] = {}

    # KMeans
    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    kmeans_labels = kmeans.fit_predict(X)
    outputs["KMeans"] = ClusteringOutput(
        method="KMeans",
        labels=kmeans_labels,
        centers=kmeans.cluster_centers_,
        feature_cols=feature_cols,
    )

    # KMedoids
    try:
        from sklearn_extra.cluster import KMedoids

        kmedoids = KMedoids(
            n_clusters=k,
            random_state=random_state,
            metric="euclidean",
            init="k-medoids++",
        )
        kmedoids_labels = kmedoids.fit_predict(X)
        kmedoids_centers = X[kmedoids.medoid_indices_]
    except Exception:
        kmedoids_labels, kmedoids_centers = _kmedoids_fallback(
            X,
            k=k,
            random_state=random_state,
        )

    outputs["KMedoids"] = ClusteringOutput(
        method="KMedoids",
        labels=kmedoids_labels,
        centers=kmedoids_centers,
        feature_cols=feature_cols,
    )

    # Hierarchical
    hierarchical = AgglomerativeClustering(n_clusters=k, linkage="ward")
    hierarchical_labels = hierarchical.fit_predict(X)
    hierarchical_centers = _compute_centers_from_labels(X, hierarchical_labels, k)

    outputs["Hierarchical"] = ClusteringOutput(
        method="Hierarchical",
        labels=hierarchical_labels,
        centers=hierarchical_centers,
        feature_cols=feature_cols,
    )

    return outputs
