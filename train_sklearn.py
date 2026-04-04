import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split  # Chia tập train/test
from sklearn.preprocessing import StandardScaler  # Chuẩn hoá dữ liệu
from sklearn.linear_model import ElasticNet  # Model Elastic Net từ scikit-learn
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error  # Đánh giá model

# Import các hàm xử lý dữ liệu từ file khác
from preprocess import clean_data, handle_outliers_iqr
from encoding import fit_target_encoder, transform_target_encoder

# LOAD & PREPROCESSING DATA
# Load dữ liệu gốc từ file CSV
df = pd.read_csv("Data_CarPrice.csv")

# Làm sạch dữ liệu (xoá ký tự đặc biệt, chuyển đổi kiểu dữ liệu)
df = clean_data(df)

# Loại bỏ các giá trị outliers (ngoại lệ) bằng phương pháp IQR
numeric_cols = ['AskPrice', 'kmDriven', 'km_per_year']
df = handle_outliers_iqr(df, numeric_cols)

#TÁCH FEATURES & TARGET
# Tách X (features) - các cột dùng để dự đoán
# Tách y (target) - cột giá cần dự đoán
X = df.drop(columns=['AskPrice'])
y = df['AskPrice']

#TRAIN/TEST SPLIT
# Chia dữ liệu thành 80% train, 20% test
# random_state=42 để đảm bảo kết quả lặp lại được
X_train, X_test, y_train, y_test = train_test_split(
    X.values, y.values, test_size=0.2, random_state=42
)

# Chuyển lại thành DataFrame để tiện xử lý
columns_raw = X.columns
X_train = pd.DataFrame(X_train, columns=columns_raw)
X_test = pd.DataFrame(X_test, columns=columns_raw)

# TARGET ENCODING
# Mục đích: Chuyển các cột categorical (Brand, model) thành số dựa vào trung bình target
# NO LEAK: Chỉ fit trên train data, áp dụng lên test data
global_mean = y_train.mean()  # Trung bình giá trên tập train
target_maps = {}  # Lưu mapping từ giá trị categorical -> trung bình giá

# Encoding cho 2 cột: Brand và model
for col in ['Brand', 'model']:
    if col in X_train.columns:
        # Tạo DataFrame tạm để fit encoder
        train_df = X_train.copy()
        train_df['AskPrice'] = y_train

        # Tính toán ánh xạ target (giá trị categorical -> trung bình giá)
        mean_map = fit_target_encoder(train_df, col, 'AskPrice')
        target_maps[col] = mean_map

        # Áp dụng ánh xạ lên train và test
        X_train[col] = transform_target_encoder(X_train, col, mean_map, global_mean)
        X_test[col] = transform_target_encoder(X_test, col, mean_map, global_mean)

# ONE-HOT ENCODING
# Chuyển các cột categorical còn lại thành dạng binary (0/1)
# drop_first=True: Loại bỏ cột đầu tiên để tránh multicollinearity
X_train = pd.get_dummies(X_train, drop_first=True)
X_test = pd.get_dummies(X_test, drop_first=True)

# ALIGN COLUMNS
# QUAN TRỌNG: Đảm bảo train và test có cùng số lượng và thứ tự cột
# Trong quá trình one-hot encoding, test có thể thiếu một số cột (nếu không có giá trị đó)
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

# Lưu danh sách cột cuối cùng (dùng khi predict trên dữ liệu mới)
columns_final = X_train.columns.tolist()

#FIX DATA TYPES
# Đảm bảo tất cả dữ liệu là số (float64) để tránh lỗi khi train model
X_train = X_train.apply(pd.to_numeric, errors='coerce')  # Chuyển sang số, lỗi -> NaN
X_test = X_test.apply(pd.to_numeric, errors='coerce')

# Thay NaN thành 0
X_train = X_train.fillna(0)
X_test = X_test.fillna(0)

# CONVERT TO NUMPY
# Chuyển sang numpy array và đảm bảo kiểu float64
X_train = X_train.values.astype(np.float64)
X_test = X_test.values.astype(np.float64)

# STANDARDIZATION (Z-SCORE NORMALIZATION)
# Mục đích: Chuẩn hoá dữ liệu sao cho trung bình = 0, độ lệch chuẩn = 1
# Điều này giúp model hội tụ nhanh hơn
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  # Fit trên train data
X_test = scaler.transform(X_test)  # Áp dụng lên test (không fit lại)

#TRAIN ELASTIC NET MODEL
# Elastic Net = kết hợp L1 (Lasso) + L2 (Ridge) regularization
alpha = 0.1  # Mức độ phạt regularization (càng cao càng mạnh)
l1_ratio = 0.5  # 0.5 = 50% L1 + 50% L2 (cân bằng giữa 2 loại)

# Tạo và train model scikit-learn Elastic Net
model_sklearn = ElasticNet(
    alpha=alpha,
    l1_ratio=l1_ratio,
    max_iter=1000,  # Số vòng lặp tối đa
    random_state=42,
    verbose=1  # In log trong quá trình training
)

model_sklearn.fit(X_train, y_train)  # Train model trên tập training

# SAVE MODEL
# Lưu toàn bộ pipeline (model + scaler + metadata) để dùng lại sau
model = {
    "model": model_sklearn,  # Model Elastic Net đã train
    "scaler": scaler,  # Scaler để chuẩn hoá dữ liệu mới
    "columns": columns_final,  # Danh sách cột sau encoding (để align dữ liệu mới)
    "target_maps": target_maps,  # Ánh xạ target encoding cho Brand, model
    "global_mean": global_mean,  # Trung bình giá (dùng cho target encoding)
    "alpha": alpha,  # Tham số regularization
    "l1_ratio": l1_ratio  # Tỷ lệ L1/L2
}

# Lưu xuống file pickle
with open("elastic_model_sklearn.pkl", "wb") as f:
    pickle.dump(model, f)

print("Đã lưu model vào elastic_model_sklearn.pkl")

# EVALUATE MODEL==
# Dự đoán giá trên tập test
y_pred = model_sklearn.predict(X_test)

# Tính các chỉ số đánh giá
r2 = r2_score(y_test, y_pred)  # R² Score: 1 = hoàn hảo, 0 = kém
rmse = np.sqrt(mean_squared_error(y_test, y_pred))  # RMSE: sai số bình phương trung bình
mae = mean_absolute_error(y_test, y_pred)  # MAE: sai số tuyệt đối trung bình

# In kết quả
print(f"\n=== KÊTS QUẢ SCIKIT-LEARN ===")
print(f"R² Score: {r2:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"Số lượng features: {len(columns_final)}")
print(f"Số non-zero coefficients: {np.sum(model_sklearn.coef_ != 0)}")
