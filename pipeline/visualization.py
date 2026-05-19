"""visualization.py

Các hàm vẽ biểu đồ cho giao diện Tkinter.
"""
from __future__ import annotations

from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pipeline.clustering import ClusteringOutput


def visualize_clusters(
    labeled_df: pd.DataFrame,
    output: ClusteringOutput,
    scaler,
    max_plot_points: int = 10000,
):
    """Scatter plot: x = age, y = totalprice; hiển thị centroid/medoid.

    Nếu dữ liệu quá lớn, chỉ vẽ mẫu để tránh treo GUI/tràn RAM.
    """
    fig, ax = plt.subplots(figsize=(9.5, 6))

    plot_df = labeled_df
    sampled = False
    if len(plot_df) > max_plot_points:
        plot_df = plot_df.sample(n=max_plot_points, random_state=42)
        sampled = True

    for label, sub in plot_df.groupby("label"):
        ax.scatter(
            sub["age"],
            sub["totalprice"],
            s=35,
            alpha=0.70,
            label=label,
        )

    centers_original = scaler.inverse_transform(output.centers)
    marker = "*" if output.method == "KMeans" else "D"
    center_name = "Centroid" if output.method == "KMeans" else "Medoid"

    ax.scatter(
        centers_original[:, 0],
        centers_original[:, 1],
        s=280,
        marker=marker,
        edgecolors="black",
        linewidths=1.2,
        label=f"{center_name} - {output.method}",
    )

    for i, (age, totalprice) in enumerate(centers_original):
        ax.annotate(
            f"C{i}",
            (age, totalprice),
            textcoords="offset points",
            xytext=(8, 8),
            fontsize=9,
            weight="bold",
        )

    title = f"Phân cụm khách hàng bằng {output.method} - k={output.k}"
    if sampled:
        title += f" — vẽ mẫu {max_plot_points:,}/{len(labeled_df):,} khách hàng"
    ax.set_title(title)
    ax.set_xlabel("age")
    ax.set_ylabel("totalprice")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)

    return fig


def plot_elbow(evaluation_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9.5, 6))

    kmeans_df = evaluation_df[evaluation_df["method"] == "KMeans"].sort_values("k")
    ax.plot(kmeans_df["k"], kmeans_df["inertia_wcss"], marker="o")
    ax.set_title("Elbow Method cho KMeans")
    ax.set_xlabel("Số cụm k")
    ax.set_ylabel("Inertia / WCSS")
    ax.grid(True, alpha=0.3)

    return fig


def plot_silhouette(evaluation_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9.5, 6))

    for method in ["KMeans", "KMedoids"]:
        sub = evaluation_df[evaluation_df["method"] == method].sort_values("k")
        if sub.empty:
            continue
        ax.plot(sub["k"], sub["silhouette_score"], marker="o", label=method)

    ax.set_title("So sánh Silhouette Score theo từng k")
    ax.set_xlabel("Số cụm k")
    ax.set_ylabel("Silhouette Score")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    return fig
