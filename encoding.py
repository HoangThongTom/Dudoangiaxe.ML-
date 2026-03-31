import pandas as pd
import numpy as np

def encode_data(file_path):
    # Đọc dữ liệu đã sạch từ Bước 1
    df = pd.read_excel(file_path)
    
    # One-Hot Encoding (cho các cột ít giá trị: Transmission, Owner, FuelType)
    # pd.get_dummies thuộc pandas, không cần sklearn
    cat_cols = ['Transmission', 'Owner', 'FuelType']
    df = pd.get_dummies(df, columns=[c for c in cat_cols if c in df.columns], drop_first=True)
    
    # Target Encoding thủ công (cho các cột nhiều giá trị: Brand, Model)
    for col in ['Brand', 'model']:
        if col in df.columns:
            
            mean_map = df.groupby(col)['AskPrice'].mean()# Tính giá trung bình cho mỗi loại
            df[f'{col}_encoded'] = df[col].map(mean_map)# Map giá trị đó vào cột
            df = df.drop(col, axis=1)# Xóa cột chữ gốc
    
    # Chuẩn hóa phân phối Target (AskPrice) bằng Log Transform
    df['AskPrice'] = np.log1p(df['AskPrice'])
    
    return df

if __name__ == "__main__":
    path = "Data_CarPrice.xlsx"
    df_final = encode_data(path) # Tạo DataFrame cuối cùng cho ML
    
    print("Dữ liệu đã mã hóa hoàn toàn thành số.")
    print(df_final.head())
    
    df_final.to_csv("Data_Ready_For_ML.csv", index=False)# Lưu ra file csv riêng cho ML (vì excel có thể làm sai lệch định dạng số lớn)