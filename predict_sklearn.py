import pandas as pd
import numpy as np
import pickle
from preprocess import clean_data, handle_outliers_iqr
from encoding import transform_target_encoder

# Load lại toàn bộ "bộ não" đã lưu ở bước Train
def load_sklearn_model(model_path="elastic_model_sklearn.pkl"):
    with open(model_path, "rb") as f:
        return pickle.load(f)

# Hàm xử lý dữ liệu mới và dự đoán
def preprocess_and_predict(new_data, model_dict):
    # Trích xuất các tham số đã lưu
    model = model_dict["model"]
    scaler = model_dict["scaler"]
    columns_final = model_dict["columns"]
    target_maps = model_dict["target_maps"]
    global_mean = model_dict["global_mean"]

    # Đọc dữ liệu (từ đường dẫn file hoặc DataFrame)
    if isinstance(new_data, str):
        X = pd.read_csv(new_data)
    else:
        X = new_data.copy()

    # Làm sạch giống hệt lúc Train
    X = clean_data(X)
    numeric_cols = ['kmDriven', 'km_per_year']
    X = handle_outliers_iqr(X, numeric_cols)

    # Encode bằng TỪ ĐIỂN CŨ (target_maps) của lúc Train
    for col in ['Brand', 'model']:
        if col in X.columns and col in target_maps:
            mean_map = target_maps[col]
            X[col] = transform_target_encoder(X, col, mean_map, global_mean)

    # One-hot và ép số lượng cột khớp 100% với lúc Train (cột nào mới xuất hiện thì bỏ, cột nào thiếu thì điền 0)
    X = pd.get_dummies(X, drop_first=True)
    X = X.reindex(columns=columns_final, fill_value=0)
    
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
    X_np = X.values.astype(np.float64)

    # Chuẩn hóa (Scale) dữ liệu bằng SCALER CŨ
    X_scaled = scaler.transform(X_np)

    # Gọi hàm predict của scikit-learn để đưa ra kết quả
    predictions = model.predict(X_scaled)
    return predictions

if __name__ == "__main__":
    model_dict = load_sklearn_model("elastic_model_sklearn.pkl")
    df_test = pd.read_csv("Data_CarPrice.csv") # Ở đây có thể thay bằng file chứa xe mới chưa có giá
    
    predictions = preprocess_and_predict(df_test, model_dict)
    
    print(f"Dự đoán {len(predictions)} mẫu")
    print(f"Giá trung bình dự đoán: {predictions.mean():.2f}")
    print(f"Min: {predictions.min():.2f}, Max: {predictions.max():.2f}")