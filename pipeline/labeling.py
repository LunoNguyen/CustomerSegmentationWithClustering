"""labeling.py

Gán nhãn khách hàng dựa trên đặc điểm cụm sau khi gom cụm.
Không gán nhãn cứng trước clustering.

Nhãn được xếp theo giá trị cụm từ thấp đến cao:
1. Khách hàng vãng lai
2. Khách hàng mới
3. Khách hàng thân thiết
4. Khách hàng trung thành
5. Khách hàng V.I.P
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


CUSTOMER_LABELS = [
    "Khách hàng vãng lai",
    "Khách hàng mới",
    "Khách hàng thân thiết",
    "Khách hàng trung thành",
    "Khách hàng V.I.P",
]


def _scale_series(values: pd.Series) -> pd.Series:
    if len(values) == 1 or values.nunique(dropna=True) <= 1:
        return pd.Series(np.ones(len(values)), index=values.index)
    scaler = MinMaxScaler()
    return pd.Series(
        scaler.fit_transform(values.to_numpy(dtype=float).reshape(-1, 1)).ravel(),
        index=values.index,
    )


def _label_by_rank(sorted_cluster_ids: List[int]) -> Dict[int, str]:
    """Map cụm sang 5 nhãn theo thứ hạng.
    Nếu k != 5, một số nhãn có thể không xuất hiện hoặc nhiều cụm có thể chung nhãn.
    """
    k = len(sorted_cluster_ids)
    if k == 1:
        return {int(sorted_cluster_ids[0]): CUSTOMER_LABELS[2]}

    label_map = {}
    for rank, cluster_id in enumerate(sorted_cluster_ids):
        label_index = round(rank * (len(CUSTOMER_LABELS) - 1) / (k - 1))
        label_map[int(cluster_id)] = CUSTOMER_LABELS[label_index]
    return label_map


def build_cluster_statistics(
    customer_features_original: pd.DataFrame,
    labels: np.ndarray,
) -> pd.DataFrame:
    df = customer_features_original.copy()
    df["cluster"] = labels

    stats = (
        df.groupby("cluster")
        .agg(
            customer_count=("customer_id", "count"),
            avg_age=("age", "mean"),
            avg_totalprice=("totalprice", "mean"),
            min_totalprice=("totalprice", "min"),
            max_totalprice=("totalprice", "max"),
            avg_quantity=("quantity", "mean"),
            total_quantity=("quantity", "sum"),
        )
        .reset_index()
    )

    stats["cluster_size_ratio"] = stats["customer_count"] / stats["customer_count"].sum()

    total_scaled = _scale_series(stats["avg_totalprice"])
    age_scaled = _scale_series(stats["avg_age"])
    size_scaled = _scale_series(stats["cluster_size_ratio"])

    # Cụm có avg_totalprice cao là tín hiệu chính.
    # age và số lượng khách hàng trong cụm được dùng để giải thích và tinh chỉnh:
    # - age cao hơn một chút thường thể hiện nhóm ổn định hơn.
    # - cụm quá đông thường ít đặc biệt hơn nhóm VIP hiếm.
    stats["cluster_value_score"] = (
        0.75 * total_scaled
        + 0.15 * age_scaled
        - 0.10 * size_scaled
    )

    sorted_cluster_ids = (
        stats.sort_values("cluster_value_score", ascending=True)["cluster"]
        .astype(int)
        .tolist()
    )
    label_map = _label_by_rank(sorted_cluster_ids)

    stats["label"] = stats["cluster"].map(label_map)

    stats = stats[
        [
            "cluster",
            "label",
            "customer_count",
            "avg_age",
            "avg_totalprice",
            "min_totalprice",
            "max_totalprice",
            "avg_quantity",
            "total_quantity",
            "cluster_size_ratio",
            "cluster_value_score",
        ]
    ].copy()

    numeric_cols = [
        "avg_age",
        "avg_totalprice",
        "min_totalprice",
        "max_totalprice",
        "avg_quantity",
        "total_quantity",
        "cluster_size_ratio",
        "cluster_value_score",
    ]
    stats[numeric_cols] = stats[numeric_cols].round(4)

    label_order = {label: idx for idx, label in enumerate(CUSTOMER_LABELS)}
    stats["_label_order"] = stats["label"].map(label_order)
    stats = stats.sort_values("_label_order").drop(columns="_label_order")

    return stats.reset_index(drop=True)


def label_clusters(
    customer_features_original: pd.DataFrame,
    customer_features_scaled: pd.DataFrame,
    labels: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Trả về bảng khách hàng đã gán nhãn và bảng thống kê cụm."""
    cluster_stats = build_cluster_statistics(customer_features_original, labels)
    label_map = dict(zip(cluster_stats["cluster"], cluster_stats["label"]))

    result = customer_features_original.copy()
    scaled = customer_features_scaled.copy()

    result["cluster"] = labels
    result["label"] = result["cluster"].map(label_map)

    result["age_scaled"] = scaled["age"]
    result["totalprice_scaled"] = scaled["totalprice"]

    stat_cols = [
        "cluster",
        "customer_count",
        "avg_age",
        "avg_totalprice",
        "cluster_value_score",
    ]
    result = result.merge(cluster_stats[stat_cols], on="cluster", how="left")

    label_order = {label: idx for idx, label in enumerate(CUSTOMER_LABELS)}
    result["_label_order"] = result["label"].map(label_order)

    result = result.sort_values(
        by=["_label_order", "totalprice", "quantity"],
        ascending=[True, False, False],
    ).drop(columns="_label_order")

    return result.reset_index(drop=True), cluster_stats
