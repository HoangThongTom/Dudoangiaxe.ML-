import pandas as pd
import numpy as np

def encode_data(file_path):
    df = pd.read_csv(file_path)

    cat_cols = ['Transmission', 'Owner', 'FuelType']
    # Chuyển 3 cái trên thành nhãn 0 vs 1 
    cols_to_encode = []
    for col in cat_cols:
        if col in df.columns:
            cols_to_encode.append(col)
    df = pd.get_dummies(df, columns=cols_to_encode, drop_first=True, dtype=int)

    """"
    Sửa Brand_encoded và model_encoded, thử nghiệm theo hướng gom nhón 20 thương hiệu xuất hiện nhiều nhất và 'other'
    """
    # giữ Brand vì chỉ loanh quanh ít hãng 
    if 'Brand' in df.columns:
        df = pd.get_dummies(df, columns=['Brand'], drop_first=True, dtype=int)

    if 'model' in df.columns:
        model_counts = df['model'].value_counts()
        top_models = model_counts.nlargest(20).index

        new_model_column = []
        for value in df['model']:
            if value in top_models:
                new_model_column.append(value)
            else:
                new_model_column.append('other')
        df['model'] = new_model_column
        df = pd.get_dummies(df, columns=['model'], drop_first=True, dtype=int)

    # log đổi từ round(3) qua 8 coi tăng độ chính xác k        
    df['AskPrice'] = np.log1p(df['AskPrice'])
    df['AskPrice'] = df['AskPrice'].round(8)
    return df

if __name__ == "__main__":
    path = "Data_CarPrice.csv"
    try:
        df_final = encode_data(path)
        print("Data đã mã hóa hoàn toàn thành số.")
        print(df_final.head())
        
        df_final.to_csv("Data_Ready_For_ML.csv", index=False)
        print("Đã tạo file Data_Ready_For_ML.csv")
    except Exception as e:
        print(f"Lỗi khi xử lý: {e}")