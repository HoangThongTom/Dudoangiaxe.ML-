import pandas as pd
import numpy as np

def clean_data(df):
    df = df.copy()

    # Clean AskPrice (loại bỏ ký tự không phải số)
    df['AskPrice'] = df['AskPrice'].astype(str).str.replace(r'[^\d]', '', regex=True)
    df['AskPrice'] = pd.to_numeric(df['AskPrice'], errors='coerce')
    df = df.dropna(subset=['AskPrice'])  

    # Clean kmDriven
    df['kmDriven'] = df['kmDriven'].astype(str).str.replace(r'[^\d]', '', regex=True)
    df['kmDriven'] = pd.to_numeric(df['kmDriven'], errors='coerce')
    df['kmDriven'] = df['kmDriven'].fillna(df['kmDriven'].median())

    # Xử lý Age (nếu có giá trị âm hoặc vô lý)
    if (df['Age'] < 0).any():
        print("Cảnh báo: Có giá trị Age âm, sẽ thay bằng 0.")
        df['Age'] = df['Age'].clip(lower=0)

    # Feature engineering: km_per_year
    df['km_per_year'] = (df['kmDriven'] / df['Age'].replace(0, 1)).round().astype(int)

    # Xử lý ngày tháng
    df['PostedDate'] = pd.to_datetime(df['PostedDate'], errors='coerce')
    # Điền năm thiếu bằng năm trung bình của các dòng có dữ liệu
    default_year = int(df['PostedDate'].dt.year.dropna().mode()[0]) if not df['PostedDate'].dt.year.dropna().empty else 2020
    default_month = 1
    df['post_year'] = df['PostedDate'].dt.year.fillna(default_year).astype(int)
    df['post_month'] = df['PostedDate'].dt.month.fillna(default_month).astype(int)
    # Chuẩn hóa model
    df['model'] = df['model'].astype(str).str.strip().str.lower()
    df['model'] = df['model'].str.replace(r'[^a-z0-9\s-]', '', regex=True)
    df['model'] = df['model'].str.replace(r'\s+', ' ', regex=True).str.strip().str.title()

    # Drop các cột không cần thiết
    df = df.drop(['Year', 'AdditionInfo', 'PostedDate'], axis=1, errors='ignore')

    return df
# Outlier : Hàm xử lý ngoại lệ thủ công bằng IQR của Pandas
def handle_outliers_iqr(df, columns):
    df = df.copy()
    for col in columns:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
    
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)# Ép (clip) các giá trị ngoài rìa về bằng giới hạn
    return df
if __name__ == "__main__":
    path = "Data_CarPrice.xlsx"
    df_raw = pd.read_excel(path)

    # Làm sạch và lưu file kiểm tra
    df_cleaned = clean_data(df_raw)
    df_cleaned.to_excel(path, index=False) 
    print(f"Đã cập nhật dữ liệu sạch vào file {path}!")

    # XỬ lí Outlier (gọi hàm thử công)
    numeric_cols = ['AskPrice', 'kmDriven', 'km_per_year'] 
    df_cleaned = handle_outliers_iqr(df_cleaned, numeric_cols)
    df_cleaned.to_excel(path, index=False)
    print(f"Đã xử lý outliers và cập nhật dữ liệu vào file {path}!")