"""
evaluation.py

Đánh giá mô hình theo yêu cầu:
- Tổng khoảng cách từ từng điểm mall tới tâm cụm của nó.
- Tổng khoảng cách càng nhỏ thì càng tốt.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

from pipeline.clustering import ClusteringOutput


def calculate_total_distance(
    X: np.ndarray,
    labels: np.ndarray,
    centers: np.ndarray,
) -> float:
    """Tổng khoảng cách Euclidean từ từng điểm tới tâm cụm được gán."""
    assigned_centers = centers[labels]
    distances = np.linalg.norm(X - assigned_centers, axis=1)
    return float(distances.sum())


def evaluate_all_models(
    mall_features_scaled: pd.DataFrame,
    clustering_outputs: Dict[str, ClusteringOutput],
) -> Tuple[pd.DataFrame, str]:
    """Đánh giá tất cả mô hình và chọn mô hình tốt nhất."""
    rows = []

    for method, output in clustering_outputs.items():
        X = mall_features_scaled[output.feature_cols].to_numpy(dtype=float)

        total_distance = calculate_total_distance(
            X=X,
            labels=output.labels,
            centers=output.centers,
        )

        rows.append(
            {
                "method": method,
                "total_distance": total_distance,
                "note": "Tổng khoảng cách càng nhỏ càng tốt",
            }
        )

    evaluation_df = pd.DataFrame(rows)
    evaluation_df = evaluation_df.sort_values(
        by="total_distance",
        ascending=True,
    ).reset_index(drop=True)

    best_method = str(evaluation_df.loc[0, "method"])

    return evaluation_df, best_method
