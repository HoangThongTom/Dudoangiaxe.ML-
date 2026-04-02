import pandas as pd
import numpy as np

def encode_data(file_path):
    df = pd.read_csv(file_path)

    cat_cols = ['Transmission', 'Owner', 'FuelType']
    
    # One-Hot Encoding
    df = pd.get_dummies(df, columns=[c for c in cat_cols if c in df.columns], drop_first=True, dtype=int)
    global_mean = df['AskPrice'].mean() 
    
    for col in ['Brand', 'model']:
        if col in df.columns:
            mean_map = df.groupby(col)['AskPrice'].mean()
            df[f'{col}_encoded'] = df[col].map(mean_map)
            df[f'{col}_encoded'] = df[f'{col}_encoded'].fillna(global_mean)
            df[f'{col}_encoded'] = df[f'{col}_encoded'].round(3)
    
            df = df.drop(col, axis=1)
            
    # Chuẩn hóa phân phối Target (AskPrice) bằng Log Transform
    df['AskPrice'] = np.log1p(df['AskPrice'])
    df['AskPrice'] = df['AskPrice'].round(3)
    return df

if __name__ == "__main__":
    path = "Data_CarPrice.csv"
    try:
        df_final = encode_data(path)
        print("Dữ liệu đã mã hóa hoàn toàn thành số.")
        print(df_final.head())
        
        # Lưu ra CSV 
        df_final.to_csv("Data_Ready_For_ML.csv", index=False)
        print("Đã tạo file Data_Ready_For_ML.csv")
    except Exception as e:
        print(f"Lỗi khi xử lý: {e}")