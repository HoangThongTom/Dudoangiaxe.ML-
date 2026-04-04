"""
So sánh hiệu suất giữa 2 model của Scikit-learn:
1. Elastic Net (Có cơ chế Regularization chặn Overfit)
2. Linear Regression (Baseline - Hồi quy tuyến tính cơ bản)
"""
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet, LinearRegression

from preprocess import clean_data, handle_outliers_iqr
from encoding import fit_target_encoder, transform_target_encoder

# 1. PREPROCESS (Toàn bộ phần này y hệt file train_sklearn.py)
df = pd.read_csv("Data_CarPrice.csv")
df = clean_data(df)
numeric_cols = ['AskPrice', 'kmDriven', 'km_per_year']
df = handle_outliers_iqr(df, numeric_cols)

X = df.drop(columns=['AskPrice'])
y = df['AskPrice']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = pd.DataFrame(X_train, columns=X.columns).reset_index(drop=True)
X_test = pd.DataFrame(X_test, columns=X.columns).reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

global_mean = y_train.mean()
target_maps = {}
for col in ['Brand', 'model']:
    if col in X_train.columns:
        train_df = X_train.copy()
        train_df['AskPrice'] = y_train
        mean_map = fit_target_encoder(train_df, col, 'AskPrice')
        target_maps[col] = mean_map
        X_train[col] = transform_target_encoder(X_train, col, mean_map, global_mean)
        X_test[col] = transform_target_encoder(X_test, col, mean_map, global_mean)

X_train = pd.get_dummies(X_train, drop_first=True)
X_test = pd.get_dummies(X_test, drop_first=True)
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

X_train = X_train.apply(pd.to_numeric, errors='coerce').fillna(0)
X_test = X_test.apply(pd.to_numeric, errors='coerce').fillna(0)

X_train_np = X_train.values.astype(np.float64)
X_test_np = X_test.values.astype(np.float64)

# 2. SCALE BẰNG SCIKIT-LEARN
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_np)
X_test_scaled = scaler.transform(X_test_np)

# 3. TRAIN CÁC MÔ HÌNH SCIKIT-LEARN
print("=" * 60)
print("SO SÁNH CÁC MÔ HÌNH BẰNG SCIKIT-LEARN")
print("=" * 60)

# Khởi tạo và Train Mô hình 1: Elastic Net
model_en = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=1000, random_state=42)
model_en.fit(X_train_scaled, y_train)
y_pred_en = model_en.predict(X_test_scaled) # Sinh ra dự đoán của mô hình 1

# Khởi tạo và Train Mô hình 2: Linear Regression
model_lr = LinearRegression()
model_lr.fit(X_train_scaled, y_train)
y_pred_lr = model_lr.predict(X_test_scaled) # Sinh ra dự đoán của mô hình 2

# 4. ĐÁNH GIÁ & SO SÁNH
# Hàm tiện ích để tính toán nhanh 3 chỉ số quan trọng trong bài toán Regression (Hồi quy)
def evaluate(y_true, y_pred):
    return (
        r2_score(y_true, y_pred), # R2: Càng gần 1 càng tốt (Mức độ giải thích sự biến thiên của dữ liệu)
        np.sqrt(mean_squared_error(y_true, y_pred)), # RMSE: Càng thấp càng tốt (Trị tuyệt đối sai số trung bình, phạt nặng các sai số lớn)
        mean_absolute_error(y_true, y_pred) # MAE: Càng thấp càng tốt (Trung bình chênh lệch thực tế)
    )

# Gọi hàm đánh giá cho từng mô hình
r2_en, rmse_en, mae_en = evaluate(y_test, y_pred_en)
r2_lr, rmse_lr, mae_lr = evaluate(y_test, y_pred_lr)

# In kết quả
print(f"\n1. ELASTIC NET:")
print(f"   R²: {r2_en:.4f} | RMSE: {rmse_en:.4f} | MAE: {mae_en:.4f}")

print(f"\n2. LINEAR REGRESSION:")
print(f"   R²: {r2_lr:.4f} | RMSE: {rmse_lr:.4f} | MAE: {mae_lr:.4f}")

# So sánh chênh lệch hiệu suất
print("\n" + "-" * 60)
print(f"SO SÁNH (Elastic Net so với Linear Regression):")
print(f"   Khác biệt R²:   {r2_en - r2_lr:+.4f} (Lớn hơn là tốt)")
print(f"   Khác biệt RMSE: {rmse_en - rmse_lr:+.4f} (Nhỏ hơn là tốt)")
print(f"   Khác biệt MAE:  {mae_en - mae_lr:+.4f} (Nhỏ hơn là tốt)")
print("=" * 60)