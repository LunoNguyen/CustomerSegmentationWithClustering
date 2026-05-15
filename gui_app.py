"""
gui_app.py

Giao diện Tkinter cho bài toán phân cụm shopping mall.

Chức năng:
- Chọn file CSV.
- Chạy tiền xử lý, gom cụm, gán nhãn, đánh giá.
- Xem dữ liệu sau tiền xử lý.
- Chọn phương pháp gom cụm để xem biểu đồ.
- Hiển thị tâm cụm và điểm mall xung quanh tâm cụm.
- Hiển thị danh sách mall đã gán nhãn.
- Hiển thị đánh giá theo tổng khoảng cách tới tâm cụm.
"""

from __future__ import annotations

import os
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from pipeline.clustering import run_clustering_models
from pipeline.evaluation import evaluate_all_models
from pipeline.labeling import (
    build_cluster_centers_table,
    build_labeled_mall_result,
    split_malls_by_label,
)
from pipeline.preprocessing import load_and_preprocess


class MallClusteringApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Phân cụm Shopping Mall theo Mall Score")
        self.geometry("1280x780")

        self.csv_path_var = tk.StringVar(value="DataSet/customer_shopping_data.csv")
        self.method_var = tk.StringVar(value="KMeans")

        self.preprocessing_result = None
        self.clustering_outputs = {}
        self.evaluation_df = None
        self.best_method = None
        self.labeled_results = {}
        self.center_tables = {}

        self.figure = None
        self.canvas = None

        self._build_layout()

    def _build_layout(self):
        top_frame = ttk.Frame(self, padding=10)
        top_frame.pack(fill=tk.X)

        ttk.Label(top_frame, text="File CSV:").pack(side=tk.LEFT)

        path_entry = ttk.Entry(top_frame, textvariable=self.csv_path_var, width=80)
        path_entry.pack(side=tk.LEFT, padx=8)

        ttk.Button(top_frame, text="Chọn file", command=self.choose_file).pack(
            side=tk.LEFT,
            padx=4,
        )

        ttk.Button(top_frame, text="Thực thi", command=self.run_pipeline).pack(
            side=tk.LEFT,
            padx=4,
        )

        ttk.Label(top_frame, text="Phương pháp:").pack(side=tk.LEFT, padx=(20, 4))

        method_combo = ttk.Combobox(
            top_frame,
            textvariable=self.method_var,
            values=["KMeans", "KMedoids", "Hierarchical"],
            state="readonly",
            width=18,
        )
        method_combo.pack(side=tk.LEFT)
        method_combo.bind("<<ComboboxSelected>>", lambda event: self.refresh_method_views())

        self.status_var = tk.StringVar(value="Chưa chạy.")
        ttk.Label(top_frame, textvariable=self.status_var).pack(side=tk.LEFT, padx=20)

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.tab_preprocessed = ttk.Frame(self.notebook)
        self.tab_plot = ttk.Frame(self.notebook)
        self.tab_labels = ttk.Frame(self.notebook)
        self.tab_centers = ttk.Frame(self.notebook)
        self.tab_evaluation = ttk.Frame(self.notebook)

        self.notebook.add(self.tab_preprocessed, text="Dữ liệu sau tiền xử lý")
        self.notebook.add(self.tab_plot, text="Trực quan gom cụm")
        self.notebook.add(self.tab_labels, text="Danh sách gán nhãn")
        self.notebook.add(self.tab_centers, text="Tâm cụm")
        self.notebook.add(self.tab_evaluation, text="Đánh giá")

        self.preprocessed_tree = self._create_table(self.tab_preprocessed)
        self.label_tree = self._create_table(self.tab_labels)
        self.center_tree = self._create_table(self.tab_centers)
        self.evaluation_tree = self._create_table(self.tab_evaluation)

        self.plot_container = ttk.Frame(self.tab_plot)
        self.plot_container.pack(fill=tk.BOTH, expand=True)

    def _create_table(self, parent):
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.BOTH, expand=True)

        tree = ttk.Treeview(frame, show="headings")

        y_scroll = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=tree.yview)
        x_scroll = ttk.Scrollbar(frame, orient=tk.HORIZONTAL, command=tree.xview)

        tree.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)

        tree.grid(row=0, column=0, sticky="nsew")
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll.grid(row=1, column=0, sticky="ew")

        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)

        return tree

    def choose_file(self):
        path = filedialog.askopenfilename(
            title="Chọn file CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if path:
            self.csv_path_var.set(path)

    def run_pipeline(self):
        csv_path = self.csv_path_var.get().strip()

        if not csv_path:
            messagebox.showerror("Lỗi", "Vui lòng chọn file CSV.")
            return

        if not os.path.exists(csv_path):
            messagebox.showerror("Lỗi", f"Không tìm thấy file:\n{csv_path}")
            return

        try:
            self.status_var.set("Đang xử lý...")
            self.update_idletasks()

            self.preprocessing_result = load_and_preprocess(
                csv_path=csv_path,
                outlier_method="iqr",
                iqr_multiplier=1.5,
            )

            self.clustering_outputs = run_clustering_models(
                mall_features_scaled=self.preprocessing_result.mall_features_scaled,
                feature_cols=self.preprocessing_result.feature_cols,
                k=3,
                random_state=42,
            )

            self.evaluation_df, self.best_method = evaluate_all_models(
                mall_features_scaled=self.preprocessing_result.mall_features_scaled,
                clustering_outputs=self.clustering_outputs,
            )

            self.labeled_results = {}
            self.center_tables = {}

            for method, output in self.clustering_outputs.items():
                labeled = build_labeled_mall_result(
                    mall_features_original=self.preprocessing_result.mall_features_original,
                    mall_features_scaled=self.preprocessing_result.mall_features_scaled,
                    labels=output.labels,
                    centers=output.centers,
                    feature_cols=output.feature_cols,
                )
                centers_table = build_cluster_centers_table(
                    centers=output.centers,
                    feature_cols=output.feature_cols,
                )

                self.labeled_results[method] = labeled
                self.center_tables[method] = centers_table

            self.method_var.set(self.best_method)

            self._save_outputs()
            self._display_preprocessed_data()
            self._display_evaluation()
            self.refresh_method_views()

            self.status_var.set(
                f"Hoàn tất. Mô hình tốt nhất: {self.best_method}. "
                f"Outlier đã loại: {self.preprocessing_result.outlier_count}."
            )

        except Exception as exc:
            messagebox.showerror("Lỗi khi chạy chương trình", str(exc))
            self.status_var.set("Lỗi.")

    def _save_outputs(self):
        output_dir = Path("outputs")
        output_dir.mkdir(parents=True, exist_ok=True)

        self.preprocessing_result.row_level_data.to_csv(
            output_dir / "row_level_preprocessed.csv",
            index=False,
            encoding="utf-8-sig",
        )
        self.preprocessing_result.mall_features_original.to_csv(
            output_dir / "mall_features_original.csv",
            index=False,
            encoding="utf-8-sig",
        )
        self.preprocessing_result.mall_features_scaled.to_csv(
            output_dir / "mall_features_scaled.csv",
            index=False,
            encoding="utf-8-sig",
        )
        self.evaluation_df.to_csv(
            output_dir / "model_evaluation_total_distance.csv",
            index=False,
            encoding="utf-8-sig",
        )

        for method, df in self.labeled_results.items():
            safe_name = method.lower().replace(" ", "_")
            df.to_csv(
                output_dir / f"{safe_name}_labeled_mall_result.csv",
                index=False,
                encoding="utf-8-sig",
            )

        best_result = self.labeled_results[self.best_method]
        best_result.to_csv(
            output_dir / "best_clustering_result.csv",
            index=False,
            encoding="utf-8-sig",
        )

        mall_groups = split_malls_by_label(best_result)
        mall_groups["mall_tiem_nang_cao"].to_csv(
            output_dir / "mall_tiem_nang_cao.csv",
            index=False,
            encoding="utf-8-sig",
        )
        mall_groups["mall_trung_binh"].to_csv(
            output_dir / "mall_trung_binh.csv",
            index=False,
            encoding="utf-8-sig",
        )
        mall_groups["mall_chi_tieu_thap"].to_csv(
            output_dir / "mall_chi_tieu_thap.csv",
            index=False,
            encoding="utf-8-sig",
        )

    def refresh_method_views(self):
        if not self.labeled_results:
            return

        method = self.method_var.get()
        self._display_labeled_result(method)
        self._display_centers(method)
        self._draw_cluster_plot(method)

    def _format_value(self, value):
        if isinstance(value, float):
            return f"{value:.6f}".rstrip("0").rstrip(".")
        return value

    def _display_dataframe(self, tree, df: pd.DataFrame, max_rows: int | None = None):
        tree.delete(*tree.get_children())

        if df is None or df.empty:
            tree["columns"] = []
            return

        display_df = df.copy()

        if max_rows is not None:
            display_df = display_df.head(max_rows)

        columns = list(display_df.columns)
        tree["columns"] = columns

        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=max(120, min(260, len(col) * 12)), anchor=tk.CENTER)

        for _, row in display_df.iterrows():
            values = [self._format_value(row[col]) for col in columns]
            tree.insert("", tk.END, values=values)

    def _display_preprocessed_data(self):
        df = self.preprocessing_result.mall_features_original.copy()
        scaled = self.preprocessing_result.mall_features_scaled.copy()

        for col in self.preprocessing_result.feature_cols:
            df[f"{col}_scaled"] = scaled[col]

        self._display_dataframe(self.preprocessed_tree, df)

    def _display_labeled_result(self, method: str):
        df = self.labeled_results[method].copy()

        preferred_cols = [
            "shopping_mall",
            "label",
            "cluster",
            "mall_score",
            "total_spending",
            "avg_spending",
            "median_spending",
            "max_spending",
            "transaction_count",
            "high_value_ratio",
            "distance_to_center",
        ]

        cols = [col for col in preferred_cols if col in df.columns]
        self._display_dataframe(self.label_tree, df[cols])

    def _display_centers(self, method: str):
        self._display_dataframe(self.center_tree, self.center_tables[method])

    def _display_evaluation(self):
        self._display_dataframe(self.evaluation_tree, self.evaluation_df)

    def _draw_cluster_plot(self, method: str):
        for child in self.plot_container.winfo_children():
            child.destroy()

        result = self.labeled_results[method].copy()
        centers_table = self.center_tables[method].copy()

        # Vẽ theo 2 trục:
        # x = transaction_count_scaled
        # y = mall_score_scaled
        x_col = "transaction_count_scaled"
        y_col = "mall_score_scaled"

        result = result.sort_values("mall_score", ascending=True).reset_index(drop=True)

        fig, ax = plt.subplots(figsize=(10, 6))

        labels = ["Thấp", "Trung bình", "Cao"]

        for label in labels:
            sub = result[result["label"] == label]
            if sub.empty:
                continue
            ax.scatter(
                sub[x_col],
                sub[y_col],
                s=90,
                label=f"Điểm mall - {label}",
            )

            for _, row in sub.iterrows():
                ax.annotate(
                    row["shopping_mall"],
                    (row[x_col], row[y_col]),
                    textcoords="offset points",
                    xytext=(5, 5),
                    fontsize=8,
                )

        # Vẽ tâm cụm.
        for _, center_row in centers_table.iterrows():
            cx = center_row["center_transaction_count"]
            cy = center_row["center_mall_score"]
            cluster_label = center_row["label"]

            ax.scatter(
                cx,
                cy,
                s=300,
                marker="*",
                edgecolors="black",
                linewidths=1.2,
                label=f"Tâm cụm - {cluster_label}",
            )

            # Nối điểm tới tâm cụm.
            cluster_id = center_row["cluster"]
            cluster_points = result[result["cluster"] == cluster_id]
            for _, point in cluster_points.iterrows():
                ax.plot(
                    [point[x_col], cx],
                    [point[y_col], cy],
                    linestyle="--",
                    linewidth=0.8,
                    alpha=0.6,
                )

        ax.set_title(f"Gom cụm mall bằng {method}")
        ax.set_xlabel("transaction_count_scaled")
        ax.set_ylabel("mall_score_scaled")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)

        self.figure = fig
        self.canvas = FigureCanvasTkAgg(fig, master=self.plot_container)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
