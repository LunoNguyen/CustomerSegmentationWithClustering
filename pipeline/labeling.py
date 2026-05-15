"""
labeling.py

Gán nhãn cụm Cao / Trung bình / Thấp dựa trên mall_score.

Cách làm:
- Sau khi gom cụm, mỗi cụm có một tâm.
- Xác định vị trí cột mall_score trong feature_cols.
- Cụm có tâm mall_score cao nhất => Cao.
- Cụm có tâm mall_score thấp nhất => Thấp.
- Cụm còn lại => Trung bình.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd


def build_cluster_label_map(
    centers: np.ndarray,
    feature_cols: List[str],
) -> Dict[int, str]:
    """Tạo mapping cluster_id -> label dựa trên mall_score của tâm cụm."""
    if "mall_score" not in feature_cols:
        raise ValueError("feature_cols phải có cột mall_score để gán nhãn.")

    score_index = feature_cols.index("mall_score")
    center_scores = centers[:, score_index]

    sorted_cluster_ids = np.argsort(center_scores)

    label_map = {
        int(sorted_cluster_ids[0]): "Thấp",
        int(sorted_cluster_ids[1]): "Trung bình",
        int(sorted_cluster_ids[2]): "Cao",
    }

    return label_map


def build_labeled_mall_result(
    mall_features_original: pd.DataFrame,
    mall_features_scaled: pd.DataFrame,
    labels: np.ndarray,
    centers: np.ndarray,
    feature_cols: List[str],
) -> pd.DataFrame:
    """Gán cluster và label cuối cùng cho từng mall."""
    result = mall_features_original.copy()
    scaled_part = mall_features_scaled.copy()

    result["cluster"] = labels
    scaled_part["cluster"] = labels

    label_map = build_cluster_label_map(centers, feature_cols)
    result["label"] = result["cluster"].map(label_map)

    # Thêm các cột scaled để kiểm tra.
    for col in feature_cols:
        result[f"{col}_scaled"] = scaled_part[col]

    # Khoảng cách từ từng mall tới tâm cụm của nó.
    X = mall_features_scaled[feature_cols].to_numpy(dtype=float)
    assigned_centers = centers[labels]
    result["distance_to_center"] = np.linalg.norm(X - assigned_centers, axis=1)

    # Sắp xếp theo nhãn và điểm.
    label_order = {"Cao": 0, "Trung bình": 1, "Thấp": 2}
    result["_label_order"] = result["label"].map(label_order)
    result = result.sort_values(
        by=["_label_order", "mall_score", "total_spending"],
        ascending=[True, False, False],
    ).drop(columns=["_label_order"])

    return result.reset_index(drop=True)


def build_cluster_centers_table(
    centers: np.ndarray,
    feature_cols: List[str],
) -> pd.DataFrame:
    """Tạo bảng tâm cụm để hiển thị lên giao diện."""
    label_map = build_cluster_label_map(centers, feature_cols)

    rows = []
    for cluster_id, center in enumerate(centers):
        row = {"cluster": cluster_id, "label": label_map[cluster_id]}
        for col, value in zip(feature_cols, center):
            row[f"center_{col}"] = value
        rows.append(row)

    df = pd.DataFrame(rows)

    label_order = {"Cao": 0, "Trung bình": 1, "Thấp": 2}
    df["_label_order"] = df["label"].map(label_order)
    df = df.sort_values("_label_order").drop(columns="_label_order")

    return df.reset_index(drop=True)


def split_malls_by_label(result: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Tách mall theo nhãn."""
    return {
        "mall_tiem_nang_cao": result[result["label"] == "Cao"].copy(),
        "mall_trung_binh": result[result["label"] == "Trung bình"].copy(),
        "mall_chi_tieu_thap": result[result["label"] == "Thấp"].copy(),
    }
