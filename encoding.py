import pandas as pd
import numpy as np

def fit_one_hot_encoder(df, categorical_cols):
    df_encoded = pd.get_dummies(df, columns=categorical_cols, dtype=int)
    return df_encoded

def encode_data_legacy(file_path):
    df = pd.read_csv(file_path)
    categorical_features = ['Brand', 'model', 'Transmission', 'Owner', 'FuelType']
    existing_cat_cols = [col for col in categorical_features if col in df.columns]
    df_final = fit_one_hot_encoder(df, existing_cat_cols)
    return df_final

if __name__ == "__main__":
    path = "Data_CarPrice.csv"
    try:
        df_final = encode_data_legacy(path)
        print("Dữ liệu đã mã hóa hoàn toàn thành số.")
        print(df_final.head())
        df_final.to_csv("Data_Ready_For_ML.csv", index=False)
        print("Đã tạo file Data_Ready_For_ML.csv")
    except Exception as e:
        print(f"Lỗi khi xử lý: {e}")