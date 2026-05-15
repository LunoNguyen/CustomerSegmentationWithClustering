# Mall Spending Clustering GUI

Chương trình phân loại shopping mall thành 3 nhóm:

- Cao
- Trung bình
- Thấp

Phiên bản này dùng cách hợp lý hơn majority vote:

```text
Tổng hợp đặc trưng theo từng mall
→ Scale MinMax
→ Tính mall_score có transaction_count
→ Gom cụm mall
→ Gán nhãn cụm theo mall_score của tâm cụm
```

## Cấu trúc thư mục

```text
BTL/
├── main.py
├── gui_app.py
├── requirements.txt
├── DataSet/
│   └── customer_shopping_data.csv
├── pipeline/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── clustering.py
│   ├── labeling.py
│   └── evaluation.py
└── outputs/
```

## Cách chạy

```bash
pip install -r requirements.txt
python main.py
```

## Công thức mall_score

```text
mall_score =
    0.40 * total_spending_scaled
  + 0.20 * avg_spending_scaled
  + 0.15 * median_spending_scaled
  + 0.15 * transaction_count_scaled
  + 0.10 * high_value_ratio_scaled
```

Trong đó:

| Cột | Ý nghĩa |
|---|---|
| total_spending | Tổng chi tiêu của mall |
| avg_spending | Chi tiêu trung bình mỗi giao dịch |
| median_spending | Trung vị chi tiêu |
| transaction_count | Số lượng giao dịch của mall |
| high_value_ratio | Tỷ lệ giao dịch có giá trị cao |

## Đánh giá mô hình

Mô hình được đánh giá bằng:

```text
Tổng khoảng cách từ từng mall tới tâm cụm của nó
```

Tổng khoảng cách càng nhỏ thì mô hình càng tốt.

## Output

Sau khi chạy, kết quả nằm trong thư mục `outputs/`.
