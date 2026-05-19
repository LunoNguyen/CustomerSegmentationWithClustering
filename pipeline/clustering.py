"""clustering.py

Chạy KMeans và KMedoids cho nhiều giá trị k.
Bản tối ưu RAM:
- KMeans dùng thuật toán lloyd, n_init thấp hơn để giảm thời gian.
- KMedoids được huấn luyện trên mẫu nếu số khách hàng quá lớn, sau đó gán toàn bộ khách hàng vào medoid gần nhất.
- Không tạo ma trận khoảng cách n x n cho toàn bộ dữ liệu.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances, pairwise_distances_argmin


@dataclass
class ClusteringOutput:
    method: str
    k: int
    labels: np.ndarray
    centers: np.ndarray
    feature_cols: List[str]
    inertia: Optional[float] = None
    medoid_indices: Optional[np.ndarray] = None
    sampled_for_fit: bool = False
    fit_sample_size: Optional[int] = None


def _sample_indices(n_samples: int, max_samples: int, random_state: int) -> np.ndarray:
    if n_samples <= max_samples:
        return np.arange(n_samples)
    rng = np.random.default_rng(random_state)
    return np.sort(rng.choice(n_samples, size=max_samples, replace=False))


def _pam_on_sample(
    X_sample: np.ndarray,
    k: int,
    random_state: int = 42,
    max_iter: int = 40,
) -> tuple[np.ndarray, np.ndarray, float]:
    """PAM/KMedoids fallback chỉ chạy trên mẫu để tránh tràn RAM."""
    rng = np.random.default_rng(random_state)
    n = len(X_sample)
    if n < k:
        raise ValueError("Số điểm nhỏ hơn số cụm k.")

    medoid_indices = rng.choice(n, size=k, replace=False)

    for _ in range(max_iter):
        distances_to_medoids = pairwise_distances(X_sample, X_sample[medoid_indices])
        labels = distances_to_medoids.argmin(axis=1)

        new_medoid_indices = medoid_indices.copy()
        for cluster_id in range(k):
            cluster_indices = np.where(labels == cluster_id)[0]
            if len(cluster_indices) == 0:
                continue

            # Chỉ tạo ma trận trong cụm của MẪU, không phải toàn bộ dữ liệu.
            cluster_distances = pairwise_distances(X_sample[cluster_indices])
            total_distances = cluster_distances.sum(axis=1)
            new_medoid_indices[cluster_id] = cluster_indices[total_distances.argmin()]

        if np.array_equal(medoid_indices, new_medoid_indices):
            break
        medoid_indices = new_medoid_indices

    sample_distances = pairwise_distances(X_sample, X_sample[medoid_indices])
    sample_labels = sample_distances.argmin(axis=1)
    inertia_like = float(np.sum(sample_distances.min(axis=1) ** 2))
    return sample_labels, medoid_indices, inertia_like


def run_kmeans(
    customer_features_scaled: pd.DataFrame,
    feature_cols: List[str],
    k: int,
    random_state: int = 42,
) -> ClusteringOutput:
    X = customer_features_scaled[feature_cols].to_numpy(dtype=np.float32)

    model = KMeans(
        n_clusters=k,
        random_state=random_state,
        n_init=5,
        algorithm="lloyd",
    )
    labels = model.fit_predict(X)

    return ClusteringOutput(
        method="KMeans",
        k=k,
        labels=labels.astype(int),
        centers=model.cluster_centers_,
        feature_cols=feature_cols,
        inertia=float(model.inertia_),
    )


def run_kmedoids(
    customer_features_scaled: pd.DataFrame,
    feature_cols: List[str],
    k: int,
    random_state: int = 42,
    max_fit_samples: int = 5000,
) -> ClusteringOutput:
    """Chạy KMedoids theo cách tiết kiệm RAM.

    KMedoids/PAM chuẩn thường cần ma trận khoảng cách n x n, rất dễ tràn RAM
    với dữ liệu nhiều khách hàng. Vì vậy hàm này huấn luyện medoid trên tối đa
    max_fit_samples khách hàng, rồi gán toàn bộ khách hàng vào medoid gần nhất.
    """
    X = customer_features_scaled[feature_cols].to_numpy(dtype=np.float32)
    n_samples = len(X)
    fit_indices = _sample_indices(n_samples, max_fit_samples, random_state)
    X_fit = X[fit_indices]
    sampled_for_fit = n_samples > max_fit_samples

    try:
        from sklearn_extra.cluster import KMedoids

        model = KMedoids(
            n_clusters=k,
            random_state=random_state,
            metric="euclidean",
            init="k-medoids++",
            max_iter=100,
        )
        fit_labels = model.fit_predict(X_fit)
        fit_medoid_indices = model.medoid_indices_
        inertia_like = float(model.inertia_)
    except Exception:
        fit_labels, fit_medoid_indices, inertia_like = _pam_on_sample(
            X_sample=X_fit,
            k=k,
            random_state=random_state,
        )

    global_medoid_indices = fit_indices[fit_medoid_indices]
    centers = X[global_medoid_indices]

    # Gán toàn bộ dữ liệu vào medoid gần nhất: bộ nhớ O(n*k), không phải O(n*n).
    labels = pairwise_distances_argmin(X, centers, metric="euclidean")
    distances_to_centers = pairwise_distances(X, centers, metric="euclidean")
    inertia_all = float(np.sum(distances_to_centers.min(axis=1) ** 2))

    return ClusteringOutput(
        method="KMedoids",
        k=k,
        labels=labels.astype(int),
        centers=centers,
        feature_cols=feature_cols,
        inertia=inertia_all if sampled_for_fit else inertia_like,
        medoid_indices=global_medoid_indices,
        sampled_for_fit=sampled_for_fit,
        fit_sample_size=int(len(X_fit)),
    )
