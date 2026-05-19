"""gui_app.py

Giao diện Tkinter cho bài toán phân loại khách hàng bằng KMeans và KMedoids.

Chức năng:
- Chọn file CSV.
- Tiền xử lý dữ liệu theo customer_id.
- Thử nhiều k từ 2 đến 10.
- Đánh giá KMeans/KMedoids.
- Gán nhãn khách hàng dựa trên đặc điểm cụm sau gom cụm.
- Hiển thị bảng kết quả, bảng đánh giá, elbow, silhouette, scatter plot và kết luận.
"""
from __future__ import annotations

import os
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from pipeline.evaluation import evaluate_model
from pipeline.labeling import label_clusters
from pipeline.preprocessing import preprocess_data
from pipeline.visualization import plot_elbow, plot_silhouette, visualize_clusters


class CustomerClusteringApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Phân cụm và phân loại khách hàng")
        self.geometry("1380x840")

        self.csv_path_var = tk.StringVar(value="DataSet/customer_shopping_data.csv")
        self.k_min_var = tk.StringVar(value="2")
        self.k_max_var = tk.StringVar(value="10")
        self.model_var = tk.StringVar(value="KMeans")
        self.max_table_rows = 3000

        self.preprocessing_result = None
        self.evaluation_df = None
        self.outputs = {}
        self.best_config = None
        self.labeled_results = {}
        self.cluster_stats = {}
        self.conclusion_text = ""

        self._build_layout()

    def _build_layout(self):
        top_frame = ttk.Frame(self, padding=10)
        top_frame.pack(fill=tk.X)

        ttk.Label(top_frame, text="File CSV:").pack(side=tk.LEFT)
        ttk.Entry(top_frame, textvariable=self.csv_path_var, width=70).pack(side=tk.LEFT, padx=8)

        ttk.Button(top_frame, text="Chọn file", command=self.choose_file).pack(side=tk.LEFT, padx=4)

        ttk.Label(top_frame, text="k từ").pack(side=tk.LEFT, padx=(16, 4))
        ttk.Entry(top_frame, textvariable=self.k_min_var, width=5).pack(side=tk.LEFT)
        ttk.Label(top_frame, text="đến").pack(side=tk.LEFT, padx=4)
        ttk.Entry(top_frame, textvariable=self.k_max_var, width=5).pack(side=tk.LEFT)

        ttk.Button(top_frame, text="Thực thi", command=self.run_pipeline).pack(side=tk.LEFT, padx=10)

        ttk.Label(top_frame, text="Xem mô hình:").pack(side=tk.LEFT, padx=(20, 4))
        self.model_combo = ttk.Combobox(
            top_frame,
            textvariable=self.model_var,
            values=[],
            state="readonly",
            width=22,
        )
        self.model_combo.pack(side=tk.LEFT)
        self.model_combo.bind("<<ComboboxSelected>>", lambda event: self.refresh_model_views())

        self.status_var = tk.StringVar(value="Chưa chạy.")
        ttk.Label(top_frame, textvariable=self.status_var).pack(side=tk.LEFT, padx=20)

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.tab_preprocessed = ttk.Frame(self.notebook)
        self.tab_labels = ttk.Frame(self.notebook)
        self.tab_cluster_stats = ttk.Frame(self.notebook)
        self.tab_evaluation = ttk.Frame(self.notebook)
        self.tab_elbow = ttk.Frame(self.notebook)
        self.tab_silhouette = ttk.Frame(self.notebook)
        self.tab_cluster_plot = ttk.Frame(self.notebook)
        self.tab_conclusion = ttk.Frame(self.notebook)

        self.notebook.add(self.tab_preprocessed, text="Dữ liệu khách hàng")
        self.notebook.add(self.tab_labels, text="Nhãn khách hàng")
        self.notebook.add(self.tab_cluster_stats, text="Thống kê cụm")
        self.notebook.add(self.tab_evaluation, text="Đánh giá mô hình")
        self.notebook.add(self.tab_elbow, text="Elbow KMeans")
        self.notebook.add(self.tab_silhouette, text="Silhouette")
        self.notebook.add(self.tab_cluster_plot, text="Biểu đồ phân cụm")
        self.notebook.add(self.tab_conclusion, text="Kết luận")

        self.preprocessed_tree = self._create_table(self.tab_preprocessed)
        self.label_tree = self._create_table(self.tab_labels)
        self.cluster_stats_tree = self._create_table(self.tab_cluster_stats)
        self.evaluation_tree = self._create_table(self.tab_evaluation)

        self.elbow_container = ttk.Frame(self.tab_elbow)
        self.elbow_container.pack(fill=tk.BOTH, expand=True)

        self.silhouette_container = ttk.Frame(self.tab_silhouette)
        self.silhouette_container.pack(fill=tk.BOTH, expand=True)

        self.cluster_plot_container = ttk.Frame(self.tab_cluster_plot)
        self.cluster_plot_container.pack(fill=tk.BOTH, expand=True)

        self.conclusion_box = tk.Text(self.tab_conclusion, wrap=tk.WORD, font=("Arial", 11))
        self.conclusion_box.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

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
            k_min = int(self.k_min_var.get())
            k_max = int(self.k_max_var.get())
            if k_min > k_max:
                raise ValueError("k_min phải nhỏ hơn hoặc bằng k_max.")

            self.status_var.set("Đang xử lý...")
            self.update_idletasks()

            self.preprocessing_result = preprocess_data(csv_path=csv_path)

            n_customers = len(self.preprocessing_result.customer_features_original)
            self.status_var.set(
                f"Đang đánh giá mô hình trên {n_customers:,} khách hàng... "
                "Silhouette/KMedoids sẽ dùng mẫu để tiết kiệm RAM nếu dữ liệu lớn."
            )
            self.update_idletasks()

            self.evaluation_df, self.outputs, self.best_config = evaluate_model(
                customer_features_scaled=self.preprocessing_result.customer_features_scaled,
                feature_cols=self.preprocessing_result.feature_cols,
                k_values=range(k_min, k_max + 1),
                random_state=42,
                silhouette_max_samples=5000,
                kmedoids_max_fit_samples=5000,
            )

            self.labeled_results = {}
            self.cluster_stats = {}

            for (method, k), output in self.outputs.items():
                labeled_df, stats_df = label_clusters(
                    customer_features_original=self.preprocessing_result.customer_features_original,
                    customer_features_scaled=self.preprocessing_result.customer_features_scaled,
                    labels=output.labels,
                )
                self.labeled_results[(method, k)] = labeled_df
                self.cluster_stats[(method, k)] = stats_df

            self.conclusion_text = self._build_conclusion()
            self._save_outputs()

            values = [f"{method} - k={k}" for method, k in sorted(self.outputs.keys())]
            self.model_combo["values"] = values
            self.model_var.set(f"{self.best_config[0]} - k={self.best_config[1]}")

            self._display_preprocessed_data()
            self._display_evaluation()
            self._draw_static_plots()
            self.refresh_model_views()
            self._display_conclusion()

            self.status_var.set(
                f"Hoàn tất. Mô hình đề xuất: {self.best_config[0]} - k={self.best_config[1]}. "
                f"Bảng chỉ hiển thị tối đa {self.max_table_rows:,} dòng để tránh đầy RAM; "
                "file CSV kết quả vẫn được lưu đầy đủ trong thư mục outputs/."
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
        self.preprocessing_result.customer_features_original.to_csv(
            output_dir / "customer_features_original.csv",
            index=False,
            encoding="utf-8-sig",
        )
        self.preprocessing_result.customer_features_scaled.to_csv(
            output_dir / "customer_features_scaled.csv",
            index=False,
            encoding="utf-8-sig",
        )
        self.evaluation_df.to_csv(
            output_dir / "model_evaluation_by_k.csv",
            index=False,
            encoding="utf-8-sig",
        )

        for (method, k), df in self.labeled_results.items():
            safe_name = f"{method.lower()}_k{k}"
            df.to_csv(
                output_dir / f"{safe_name}_labeled_customers.csv",
                index=False,
                encoding="utf-8-sig",
            )
            self.cluster_stats[(method, k)].to_csv(
                output_dir / f"{safe_name}_cluster_statistics.csv",
                index=False,
                encoding="utf-8-sig",
            )

        best_method, best_k = self.best_config
        self.labeled_results[(best_method, best_k)].to_csv(
            output_dir / "best_labeled_customers.csv",
            index=False,
            encoding="utf-8-sig",
        )
        self.cluster_stats[(best_method, best_k)].to_csv(
            output_dir / "best_cluster_statistics.csv",
            index=False,
            encoding="utf-8-sig",
        )

        (output_dir / "model_conclusion.txt").write_text(self.conclusion_text, encoding="utf-8")

    def refresh_model_views(self):
        if not self.labeled_results:
            return

        method, k = self._selected_method_k()
        self._display_labeled_result(method, k)
        self._display_cluster_stats(method, k)
        self._draw_cluster_plot(method, k)

    def _selected_method_k(self):
        value = self.model_var.get()
        if " - k=" not in value:
            return self.best_config

        method, k_part = value.split(" - k=")
        return method, int(k_part)

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
        if max_rows is None:
            max_rows = self.max_table_rows
        truncated = len(display_df) > max_rows
        if truncated:
            display_df = display_df.head(max_rows)

        columns = list(display_df.columns)
        tree["columns"] = columns

        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=max(120, min(280, len(col) * 13)), anchor=tk.CENTER)

        for _, row in display_df.iterrows():
            values = [self._format_value(row[col]) for col in columns]
            tree.insert("", tk.END, values=values)

        if truncated:
            values = [f"... chỉ hiển thị {max_rows:,}/{len(df):,} dòng" if i == 0 else "" for i in range(len(columns))]
            tree.insert("", tk.END, values=values)

    def _display_preprocessed_data(self):
        df = self.preprocessing_result.customer_features_original.copy()
        scaled = self.preprocessing_result.customer_features_scaled.copy()
        df["age_scaled"] = scaled["age"]
        df["totalprice_scaled"] = scaled["totalprice"]
        self._display_dataframe(self.preprocessed_tree, df)

    def _display_labeled_result(self, method: str, k: int):
        df = self.labeled_results[(method, k)].copy()
        preferred_cols = [
            "customer_id",
            "label",
            "cluster",
            "age",
            "totalprice",
            "quantity",
            "transaction_count",
            "age_scaled",
            "totalprice_scaled",
            "customer_count",
            "avg_age",
            "avg_totalprice",
            "cluster_value_score",
        ]
        cols = [col for col in preferred_cols if col in df.columns]
        self._display_dataframe(self.label_tree, df[cols])

    def _display_cluster_stats(self, method: str, k: int):
        self._display_dataframe(self.cluster_stats_tree, self.cluster_stats[(method, k)])

    def _display_evaluation(self):
        display_df = self.evaluation_df.copy()
        numeric_cols = [
            "silhouette_score",
            "davies_bouldin_index",
            "calinski_harabasz_index",
            "inertia_wcss",
            "kmedoids_distance",
            "cluster_balance_score",
            "rank_score",
        ]
        for col in numeric_cols:
            if col in display_df.columns:
                display_df[col] = display_df[col].round(6)
        self._display_dataframe(self.evaluation_tree, display_df)

    def _draw_figure_in_container(self, fig, container):
        for child in container.winfo_children():
            child.destroy()

        canvas = FigureCanvasTkAgg(fig, master=container)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _draw_static_plots(self):
        self._draw_figure_in_container(plot_elbow(self.evaluation_df), self.elbow_container)
        self._draw_figure_in_container(plot_silhouette(self.evaluation_df), self.silhouette_container)

    def _draw_cluster_plot(self, method: str, k: int):
        fig = visualize_clusters(
            labeled_df=self.labeled_results[(method, k)],
            output=self.outputs[(method, k)],
            scaler=self.preprocessing_result.scaler,
        )
        self._draw_figure_in_container(fig, self.cluster_plot_container)

    def _build_conclusion(self) -> str:
        best_method, best_k = self.best_config
        best_row = self.evaluation_df[
            (self.evaluation_df["method"] == best_method)
            & (self.evaluation_df["k"] == best_k)
        ].iloc[0]

        kmeans_best = (
            self.evaluation_df[self.evaluation_df["method"] == "KMeans"]
            .sort_values("rank_score", ascending=False)
            .iloc[0]
        )
        kmedoids_best = (
            self.evaluation_df[self.evaluation_df["method"] == "KMedoids"]
            .sort_values("rank_score", ascending=False)
            .iloc[0]
        )

        if best_method == "KMeans":
            model_reason = (
                "KMeans phù hợp hơn trong lần chạy này vì đạt điểm tổng hợp cao hơn. "
                "KMeans thường dễ giải thích bằng centroid và có Inertia/WCSS để quan sát Elbow Method."
            )
        else:
            model_reason = (
                "KMedoids phù hợp hơn trong lần chạy này vì đạt điểm tổng hợp cao hơn. "
                "KMedoids dùng medoid là khách hàng thật trong dữ liệu nên thường dễ giải thích hơn và bền hơn với điểm bất thường."
            )

        return f"""KẾT LUẬN TỰ ĐỘNG

Mô hình được đề xuất: {best_method} với k = {best_k}

Chỉ số của mô hình được đề xuất:
- Silhouette Score: {best_row["silhouette_score"]:.6f}  (càng cao càng tốt)
- Davies-Bouldin Index: {best_row["davies_bouldin_index"]:.6f}  (càng thấp càng tốt)
- Calinski-Harabasz Index: {best_row["calinski_harabasz_index"]:.6f}  (càng cao càng tốt)
- Cluster Balance Score: {best_row["cluster_balance_score"]:.6f}  (càng gần 1 càng cân bằng)
- Rank Score tổng hợp: {best_row["rank_score"]:.6f}

So sánh mô hình tốt nhất của từng thuật toán:
- KMeans tốt nhất: k={int(kmeans_best["k"])}, Silhouette={kmeans_best["silhouette_score"]:.6f}, DBI={kmeans_best["davies_bouldin_index"]:.6f}, CHI={kmeans_best["calinski_harabasz_index"]:.6f}, Balance={kmeans_best["cluster_balance_score"]:.6f}, Rank={kmeans_best["rank_score"]:.6f}
- KMedoids tốt nhất: k={int(kmedoids_best["k"])}, Silhouette={kmedoids_best["silhouette_score"]:.6f}, DBI={kmedoids_best["davies_bouldin_index"]:.6f}, CHI={kmedoids_best["calinski_harabasz_index"]:.6f}, Balance={kmedoids_best["cluster_balance_score"]:.6f}, Rank={kmedoids_best["rank_score"]:.6f}

Giải thích:
{model_reason}

Việc gán nhãn khách hàng KHÔNG được thực hiện trước khi gom cụm.
Sau khi có cụm, chương trình thống kê từng cụm theo:
- số lượng khách hàng,
- age trung bình,
- totalprice trung bình,
- totalprice nhỏ nhất,
- totalprice lớn nhất,
- điểm giá trị cụm.

Cụm có giá trị thấp hơn được gán về nhóm vãng lai/mới.
Cụm có giá trị cao hơn được gán về nhóm trung thành/V.I.P.
Nếu k khác 5, một số nhãn có thể không xuất hiện hoặc nhiều cụm có thể cùng cấp nhãn.
"""

    def _display_conclusion(self):
        self.conclusion_box.delete("1.0", tk.END)
        self.conclusion_box.insert(tk.END, self.conclusion_text)


if __name__ == "__main__":
    app = CustomerClusteringApp()
    app.mainloop()
