import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from preprocess import clean_data, handle_outliers_iqr
from encoding import fit_target_encoder, transform_target_encoder

# 1. LOAD & PREPROCESS DATA
# Đọc dữ liệu và làm sạch cơ bản (xóa null, format lại cột)
df = pd.read_csv("Data_CarPrice.csv")
df = clean_data(df)

# Xử lý ngoại lệ (Outliers) bằng phương pháp IQR để model không bị nhiễu bởi các giá trị quá bất thường
numeric_cols = ['AskPrice', 'kmDriven', 'km_per_year']
df = handle_outliers_iqr(df, numeric_cols)

# Tách biến độc lập (X) và biến mục tiêu cần dự đoán (y - Giá xe)
X = df.drop(columns=['AskPrice'])
y = df['AskPrice']

# 2. SPLIT DATA
# Chia dữ liệu: 80% để train model, 20% để test (đánh giá) model
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Chuyển về DataFrame và reset index để tránh lỗi khi merge/encode dữ liệu
X_train = pd.DataFrame(X_train, columns=X.columns).reset_index(drop=True)
X_test = pd.DataFrame(X_test, columns=X.columns).reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

# 3. ENCODING (Chuyển đổi dữ liệu chữ sang số)
global_mean = y_train.mean()
target_maps = {}

# Dùng Target Encoding cho Brand và Model (thay thế tên hãng/xe bằng giá trị trung bình của giá xe đó)
# Giúp giảm số lượng cột so với One-hot encoding khi có quá nhiều loại xe
for col in ['Brand', 'model']:
    if col in X_train.columns:
        train_df = X_train.copy()
        train_df['AskPrice'] = y_train

        # Tính toán map (từ điển mapping) chỉ trên tập Train để tránh Data Leakage (rò rỉ dữ liệu từ tập Test)
        mean_map = fit_target_encoder(train_df, col, 'AskPrice')
        target_maps[col] = mean_map

        # Áp dụng map đó cho cả tập Train và Test
        X_train[col] = transform_target_encoder(X_train, col, mean_map, global_mean)
        X_test[col] = transform_target_encoder(X_test, col, mean_map, global_mean)

# Dùng One-hot encoding cho các biến phân loại còn lại (ít giá trị rẽ nhánh hơn)
X_train = pd.get_dummies(X_train, drop_first=True)
X_test = pd.get_dummies(X_test, drop_first=True)

# Đồng bộ cột giữa Train và Test: Điền 0 nếu tập Test thiếu cột có trong tập Train (do One-hot sinh ra)
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
columns_final = X_train.columns.tolist()

# Đảm bảo toàn bộ dữ liệu đều là dạng số (numeric)
X_train = X_train.apply(pd.to_numeric, errors='coerce').fillna(0)
X_test = X_test.apply(pd.to_numeric, errors='coerce').fillna(0)

X_train_np = X_train.values.astype(np.float64)
X_test_np = X_test.values.astype(np.float64)

# 4. SCALING (Chuẩn hóa dữ liệu)
# Đưa các đặc trưng về cùng một thang đo (mean=0, std=1)
# Rất quan trọng đối với các model dựa trên khoảng cách hoặc Regularization như ElasticNet
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_np)
X_test_scaled = scaler.transform(X_test_np) # Test chỉ dùng transform, KHÔNG fit lại

# 5. TRAIN MODEL
# Huấn luyện mô hình ElasticNet (kết hợp cả L1 Ridge và L2 Lasso để giảm overfitting)
alpha = 0.1
l1_ratio = 0.5
model_sklearn = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=1000, random_state=42)
model_sklearn.fit(X_train_scaled, y_train)

# 6. SAVE MODEL
# Đóng gói toàn bộ model, bộ chuẩn hóa (scaler), danh sách cột và luật encoding vào 1 file (.pkl)
# Để khi predict file mới, ta có sẵn công thức chuẩn hóa chuẩn y như lúc train
model_data = {
    "model": model_sklearn,
    "scaler": scaler,
    "columns": columns_final,
    "target_maps": target_maps,
    "global_mean": global_mean
}

with open("elastic_model_sklearn.pkl", "wb") as f:
    pickle.dump(model_data, f)
print("Đã lưu model vào elastic_model_sklearn.pkl")

# 7. EVALUATE
# Đánh giá lại kết quả dự đoán trên tập test
y_pred = model_sklearn.predict(X_test_scaled)
print(f"\n=== KẾT QUẢ SCIKIT-LEARN ELASTIC NET ===")
print(f"R² Score: {r2_score(y_test, y_pred):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred):.4f}")