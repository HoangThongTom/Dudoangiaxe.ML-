import pandas as pd
import numpy as np
import pickle
from preprocess import clean_data, handle_outliers_iqr
from encoding import transform_target_encoder

#LOAD MODEL
def load_sklearn_model(model_path="elastic_model_sklearn.pkl"):
    """
    Load model đã train từ file pickle

    Args:
        model_path (str): Đường dẫn file model .pkl

    Returns:
        dict: Dictionary chứa model, scaler, columns, target_maps
    """
    with open(model_path, "rb") as f:
        model_dict = pickle.load(f)
    return model_dict

# PREDICT FUNCTION
def preprocess_and_predict(new_data, model_dict):
    """
    Xử lý dữ liệu mới và dự đoán giá xe

    Args:
        new_data: DataFrame hoặc đường dẫn file CSV
        model_dict: Dictionary chứa model + pipeline

    Returns:
        predictions: Array chứa dự đoán giá
    """
    # LOAD MODEL COMPONENTS
    # Lấy các thành phần từ model dictionary
    model = model_dict["model"]  # Model Elastic Net
    scaler = model_dict["scaler"]  # Scaler để chuẩn hoá
    columns_final = model_dict["columns"]  # Danh sách cột sau encoding
    target_maps = model_dict["target_maps"]  # Ánh xạ target encoding
    global_mean = model_dict["global_mean"]  # Trung bình giá cho target encoding

    # LOAD DATA
    # Load dữ liệu từ file hoặc DataFrame
    if isinstance(new_data, str):
        X = pd.read_csv(new_data)
    else:
        X = new_data.copy()

    # PREPROCESSING 
    # Làm sạch dữ liệu giống như quá trình training
    X = clean_data(X)

    # Xoá outliers
    numeric_cols = ['kmDriven', 'km_per_year']
    X = handle_outliers_iqr(X, numeric_cols)

    # TARGET ENCODING
    # Áp dụng ánh xạ target encoding đã học từ training
    for col in ['Brand', 'model']:
        if col in X.columns and col in target_maps:
            mean_map = target_maps[col]
            X[col] = transform_target_encoder(X, col, mean_map, global_mean)

    #ONE-HOT ENCODING
    # Chuyển categorical thành binary encoding
    X = pd.get_dummies(X, drop_first=True)

    # ALIGN COLUMNS
    # Đảm bảo cột của dữ liệu mới khớp với training (QUAN TRỌNG!)
    X = X.reindex(columns=columns_final, fill_value=0)

    # FIX DATA TYPES
    # Xử lý kiểu dữ liệu tương tự training
    X = X.apply(pd.to_numeric, errors='coerce')
    X = X.fillna(0)

    #CONVERT TO NUMPY & SCALE
    # Chuyển sang numpy array
    X = X.values.astype(np.float64)

    # Chuẩn hoá dữ liệu (dùng scaler từ training)
    X = scaler.transform(X)

    # PREDICT
    # Dự đoán giá
    predictions = model.predict(X)

    return predictions


# MAIN - TEST PREDICTION
if __name__ == "__main__":
    # Load model từ file pickle
    model_dict = load_sklearn_model("elastic_model_sklearn.pkl")

    # Load dữ liệu test
    df_test = pd.read_csv("Data_CarPrice.csv")

    # Dự đoán giá cho tất cả xe
    predictions = preprocess_and_predict(df_test, model_dict)

    # In kết quả
    print(f"Dự đoán {len(predictions)} mẫu")
    print(f"Giá trung bình dự đoán: {predictions.mean():.2f}")
    print(f"Min: {predictions.min():.2f}, Max: {predictions.max():.2f}")
    print(f"\n5 dự đoán đầu tiên:")
    print(predictions[:5])
