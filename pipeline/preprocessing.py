"""preprocessing.py

Tiền xử lý dữ liệu phân loại khách hàng:
- Chuẩn hóa tên cột.
- Tạo totalprice = quantity * price cho từng giao dịch.
- Xử lý missing data.
- Xử lý outlier bằng IQR capping.
- Tổng hợp dữ liệu theo customer_id.
- Chuẩn hóa age và totalprice bằng MinMaxScaler để gom cụm.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


@dataclass
class PreprocessingResult:
    original_row_count: int
    cleaned_row_count: int
    transaction_outlier_count: int
    customer_outlier_count: int
    row_level_data: pd.DataFrame
    customer_features_original: pd.DataFrame
    customer_features_scaled: pd.DataFrame
    feature_cols: List[str]
    scaler: MinMaxScaler


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(col).strip().lower().replace(" ", "_") for col in df.columns]
    return df


def _ensure_required_columns(df: pd.DataFrame) -> None:
    required = {"customer_id", "age", "price", "quantity"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            "File CSV thiếu cột bắt buộc: "
            + ", ".join(sorted(missing))
            + ". Cần có customer_id, age, price, quantity."
        )


def _fill_numeric_missing(df: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
    df = df.copy()
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        median_value = df[col].median()
        if pd.isna(median_value):
            raise ValueError(f"Cột {col} không có dữ liệu số hợp lệ.")
        df[col] = df[col].fillna(median_value)
    return df


def _cap_outliers_iqr(
    df: pd.DataFrame,
    column: str,
    iqr_multiplier: float = 1.5,
) -> Tuple[pd.DataFrame, int, float, float]:
    """Capping outlier thay vì xóa để không mất khách hàng."""
    df = df.copy()
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1

    if pd.isna(iqr) or iqr == 0:
        return df, 0, float(q1), float(q3)

    lower_bound = q1 - iqr_multiplier * iqr
    upper_bound = q3 + iqr_multiplier * iqr

    mask = (df[column] < lower_bound) | (df[column] > upper_bound)
    outlier_count = int(mask.sum())
    df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
    return df, outlier_count, float(lower_bound), float(upper_bound)


def preprocess_data(
    csv_path: str,
    iqr_multiplier: float = 1.5,
) -> PreprocessingResult:
    """Đọc CSV và tạo bảng đặc trưng theo từng customer_id."""
    df = pd.read_csv(csv_path)
    df = _normalize_columns(df)
    original_row_count = len(df)

    _ensure_required_columns(df)

    df["customer_id"] = df["customer_id"].astype(str).str.strip()
    df = df[df["customer_id"].ne("") & df["customer_id"].ne("nan")].copy()

    df = _fill_numeric_missing(df, ["age", "price", "quantity"])

    # Loại các giá trị không hợp lý sau khi đã fill missing.
    df = df[(df["age"] >= 0) & (df["price"] >= 0) & (df["quantity"] >= 0)].copy()

    df["totalprice"] = (df["quantity"] * df["price"]).round(2)

    # Xử lý outlier ở cấp giao dịch để hạn chế giao dịch bất thường chi phối tổng tiền.
    df, transaction_outlier_count, _, _ = _cap_outliers_iqr(
        df,
        column="totalprice",
        iqr_multiplier=iqr_multiplier,
    )

    cleaned_row_count = len(df)

    # Tổng hợp theo customer_id:
    # - age: lấy median để ổn định nếu một khách hàng xuất hiện nhiều giao dịch.
    # - totalprice: tổng chi tiêu.
    # - quantity: tổng số lượng sản phẩm.
    customer_features_original = (
        df.groupby("customer_id")
        .agg(
            age=("age", "median"),
            totalprice=("totalprice", "sum"),
            quantity=("quantity", "sum"),
            transaction_count=("totalprice", "count"),
        )
        .reset_index()
    )

    customer_features_original["age"] = customer_features_original["age"].round(0).astype(int)
    customer_features_original["totalprice"] = customer_features_original["totalprice"].round(2)
    customer_features_original["quantity"] = customer_features_original["quantity"].round(2)

    # Xử lý outlier cấp khách hàng trên tổng chi tiêu.
    customer_features_original, customer_outlier_count, _, _ = _cap_outliers_iqr(
        customer_features_original,
        column="totalprice",
        iqr_multiplier=iqr_multiplier,
    )
    customer_features_original["totalprice"] = customer_features_original["totalprice"].round(2)

    feature_cols = ["age", "totalprice"]

    scaler = MinMaxScaler()
    customer_features_scaled = customer_features_original.copy()
    customer_features_scaled[feature_cols] = scaler.fit_transform(
        customer_features_original[feature_cols]
    )

    return PreprocessingResult(
        original_row_count=original_row_count,
        cleaned_row_count=cleaned_row_count,
        transaction_outlier_count=transaction_outlier_count,
        customer_outlier_count=customer_outlier_count,
        row_level_data=df.reset_index(drop=True),
        customer_features_original=customer_features_original.reset_index(drop=True),
        customer_features_scaled=customer_features_scaled.reset_index(drop=True),
        feature_cols=feature_cols,
        scaler=scaler,
    )
