"""
preprocessing.py

Đọc dữ liệu CSV và tiền xử lý:
- Loại bỏ null / missing.
- Tạo totalprice nếu file chưa có cột totalprice.
- Làm tròn totalprice để tránh lỗi hiển thị số thực.
- Loại bỏ outlier bằng IQR.
- Tổng hợp đặc trưng theo từng shopping_mall.
- Scale các đặc trưng bằng MinMaxScaler.
- Tính mall_score, trong đó có transaction_count.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


@dataclass
class PreprocessingResult:
    original_row_count: int
    cleaned_row_count: int
    outlier_count: int
    row_level_data: pd.DataFrame
    mall_features_original: pd.DataFrame
    mall_features_scaled: pd.DataFrame
    feature_cols: List[str]


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Chuẩn hóa tên cột về dạng chữ thường, bỏ khoảng trắng dư."""
    df = df.copy()
    df.columns = [str(col).strip().lower().replace(" ", "_") for col in df.columns]
    return df


def _ensure_required_columns(df: pd.DataFrame) -> None:
    """Kiểm tra cột cần thiết."""
    required_base = {"shopping_mall"}

    if "totalprice" in df.columns:
        required = required_base | {"totalprice"}
    else:
        required = required_base | {"quantity", "price"}

    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            "File CSV thiếu cột bắt buộc: "
            + ", ".join(sorted(missing))
            + ". Cần có shopping_mall và totalprice, hoặc shopping_mall, quantity, price."
        )


def _create_totalprice(df: pd.DataFrame) -> pd.DataFrame:
    """Tạo hoặc chuẩn hóa cột totalprice."""
    df = df.copy()

    if "totalprice" not in df.columns:
        df["totalprice"] = df["quantity"] * df["price"]

    df["totalprice"] = pd.to_numeric(df["totalprice"], errors="coerce")
    df["totalprice"] = df["totalprice"].round(2)

    return df


def _remove_outliers_iqr(
    df: pd.DataFrame,
    column: str = "totalprice",
    iqr_multiplier: float = 1.5,
) -> Tuple[pd.DataFrame, int]:
    """Loại outlier theo IQR."""
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - iqr_multiplier * iqr
    upper_bound = q3 + iqr_multiplier * iqr

    mask = (df[column] >= lower_bound) & (df[column] <= upper_bound)
    cleaned_df = df.loc[mask].copy()
    outlier_count = int((~mask).sum())

    return cleaned_df, outlier_count


def build_mall_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Tạo đặc trưng tổng hợp theo từng mall.

    mall_score dùng công thức:
        0.40 * total_spending
      + 0.20 * avg_spending
      + 0.15 * median_spending
      + 0.15 * transaction_count
      + 0.10 * high_value_ratio

    Tất cả đặc trưng trong công thức đều đã được MinMax scale về [0, 1].
    """
    df = df.copy()

    high_threshold = df["totalprice"].quantile(0.75)
    df["is_high_value"] = df["totalprice"] >= high_threshold

    mall_features_original = df.groupby("shopping_mall").agg(
        total_spending=("totalprice", "sum"),
        avg_spending=("totalprice", "mean"),
        median_spending=("totalprice", "median"),
        max_spending=("totalprice", "max"),
        transaction_count=("totalprice", "count"),
        high_value_ratio=("is_high_value", "mean"),
    ).reset_index()

    numeric_cols = [
        "total_spending",
        "avg_spending",
        "median_spending",
        "max_spending",
        "transaction_count",
        "high_value_ratio",
    ]

    mall_features_original[numeric_cols] = mall_features_original[numeric_cols].round(4)

    scaler = MinMaxScaler()
    mall_features_scaled = mall_features_original.copy()
    mall_features_scaled[numeric_cols] = scaler.fit_transform(
        mall_features_original[numeric_cols]
    )

    # Feature dùng để clustering.
    # max_spending vẫn được giữ để mô hình nhìn được giao dịch lớn nhất,
    # nhưng mall_score không cho trọng số trực tiếp để tránh outlier chi phối quá mạnh.
    feature_cols = [
        "total_spending",
        "avg_spending",
        "median_spending",
        "max_spending",
        "transaction_count",
        "high_value_ratio",
        "mall_score",
    ]

    mall_features_scaled["mall_score"] = (
        0.40 * mall_features_scaled["total_spending"]
        + 0.20 * mall_features_scaled["avg_spending"]
        + 0.15 * mall_features_scaled["median_spending"]
        + 0.15 * mall_features_scaled["transaction_count"]
        + 0.10 * mall_features_scaled["high_value_ratio"]
    )

    mall_features_original["mall_score"] = mall_features_scaled["mall_score"].round(6)
    mall_features_scaled["mall_score"] = mall_features_scaled["mall_score"].round(6)

    return mall_features_original, mall_features_scaled, feature_cols


def load_and_preprocess(
    csv_path: str,
    outlier_method: str = "iqr",
    iqr_multiplier: float = 1.5,
) -> PreprocessingResult:
    """Đọc CSV và tiền xử lý dữ liệu."""
    df = pd.read_csv(csv_path)
    df = _normalize_columns(df)

    original_row_count = len(df)
    _ensure_required_columns(df)

    # Giữ các dòng đủ dữ liệu cần thiết.
    if "totalprice" in df.columns:
        required_for_dropna = ["shopping_mall", "totalprice"]
    else:
        required_for_dropna = ["shopping_mall", "quantity", "price"]

    df = df.dropna(subset=required_for_dropna).copy()

    # Chuẩn hóa kiểu dữ liệu.
    df["shopping_mall"] = df["shopping_mall"].astype(str).str.strip()
    df = _create_totalprice(df)
    df = df.dropna(subset=["shopping_mall", "totalprice"]).copy()
    df = df[df["totalprice"] >= 0].copy()

    # Encode string shopping_mall sang số để đáp ứng yêu cầu tiền xử lý.
    encoder = LabelEncoder()
    df["shopping_mall_encoded"] = encoder.fit_transform(df["shopping_mall"])

    # Scale row-level numeric columns.
    row_scaler = MinMaxScaler()
    df[["totalprice_scaled", "shopping_mall_encoded_scaled"]] = row_scaler.fit_transform(
        df[["totalprice", "shopping_mall_encoded"]]
    )

    # Loại outlier sau khi tạo totalprice.
    outlier_count = 0
    if outlier_method == "iqr":
        df, outlier_count = _remove_outliers_iqr(
            df,
            column="totalprice",
            iqr_multiplier=iqr_multiplier,
        )
    elif outlier_method == "none":
        outlier_count = 0
    else:
        raise ValueError("outlier_method chỉ nhận 'iqr' hoặc 'none'.")

    cleaned_row_count = len(df)

    mall_features_original, mall_features_scaled, feature_cols = build_mall_features(df)

    return PreprocessingResult(
        original_row_count=original_row_count,
        cleaned_row_count=cleaned_row_count,
        outlier_count=outlier_count,
        row_level_data=df.reset_index(drop=True),
        mall_features_original=mall_features_original,
        mall_features_scaled=mall_features_scaled,
        feature_cols=feature_cols,
    )
