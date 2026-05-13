import pandas as pd

class Preprocessing:
    def __init__(self, file_path):
        self.file_path = file_path

    def process(self):
        # Đọc dữ liệu và lấy 2 cột
        df = pd.read_csv(self.file_path)[['price', 'shopping_mall']]
        initial_count = len(df)
        
        # 1. Xóa dòng rỗng (Missing)
        df_cleaned = df.dropna()
        missing_dropped = initial_count - len(df_cleaned)
        
        # 2. Xử lý Outlier cột price (IQR)
        Q1, Q3 = df_cleaned['price'].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        df_final = df_cleaned[(df_cleaned['price'] >= Q1 - 1.5 * IQR) & (df_cleaned['price'] <= Q3 + 1.5 * IQR)].copy()
        outliers_dropped = len(df_cleaned) - len(df_final)
        
        # 3. Mã hóa shopping_mall sang số
        df_final['shopping_mall_encoded'], _ = pd.factorize(df_final['shopping_mall'])
        
        # 4. Trạng thái
        status = {
            "Tổng số dòng ban đầu": initial_count,
            "Số dòng bị xóa do rỗng (Missing)": missing_dropped,
            "Số dòng bị xóa do Outlier": outliers_dropped,
            "Tổng số dòng sau xử lý": len(df_final)
        }
        
        return df_final, status