"""
So sánh hiệu suất giữa 2 model:
1. Custom Elastic Net (Coordinate Descent implementation)
2. Scikit-learn Elastic Net

Giúp kiểm tra xem scikit-learn có tốt hơn không và các chỉ số về R², RMSE, MAE
"""
import pandas as pd
import numpy as np
import pickle
import sys
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from preprocess import clean_data, handle_outliers_iqr
from encoding import fit_target_encoder, transform_target_encoder
from elasticnet import train_test_split as custom_train_test_split
from elasticnet import StandardScaler as CustomScaler
from elasticnet import elastic_net, predict as custom_predict

#LOAD & PREPROCESS DATA
# Tất cả quá trình này giống như trong train_sklearn.py
# Load dữ liệu gốc
df = pd.read_csv("Data_CarPrice.csv")
df = clean_data(df)

# Loại bỏ outliers
numeric_cols = ['AskPrice', 'kmDriven', 'km_per_year']
df = handle_outliers_iqr(df, numeric_cols)

# Tách features và target
X = df.drop(columns=['AskPrice'])
y = df['AskPrice']

# Chia train/test bằng custom function
X_train_np, X_test_np, y_train, y_test = custom_train_test_split(X.values, y.values)

# Chuyển lại DataFrame
columns_raw = X.columns
X_train = pd.DataFrame(X_train_np, columns=columns_raw)
X_test = pd.DataFrame(X_test_np, columns=columns_raw)

# Target Encoding
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

# One-hot encoding
X_train = pd.get_dummies(X_train, drop_first=True)
X_test = pd.get_dummies(X_test, drop_first=True)

# Align columns
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

# Fix dtype
X_train = X_train.apply(pd.to_numeric, errors='coerce')
X_test = X_test.apply(pd.to_numeric, errors='coerce')

X_train = X_train.fillna(0)
X_test = X_test.fillna(0)

# Convert to numpy
X_train = X_train.values.astype(np.float64)
X_test = X_test.values.astype(np.float64)

# ============== SCALING ==============
# Dùng custom scaler để chuẩn hoá (đảm bảo dùng cùng phương pháp)
print("=" * 60)
print("CUSTOM ELASTIC NET vs SCIKIT-LEARN ELASTIC NET")
print("=" * 60)

scaler = CustomScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# TRAIN CUSTOM MODEL
print("\n1. CUSTOM ELASTIC NET (Coordinate Descent)")
print("-" * 60)
alpha = 0.1
l1_ratio = 0.5

# Train custom model
w, b = elastic_net(
    X_train_scaled,
    y_train,
    alpha=alpha,
    l1_ratio=l1_ratio,
    max_iter=1000
)

# Dự đoán với custom model
y_pred_custom = custom_predict(X_test_scaled, w, b)

# Đánh giá custom model
r2_custom = r2_score(y_test, y_pred_custom)
rmse_custom = np.sqrt(mean_squared_error(y_test, y_pred_custom))
mae_custom = mean_absolute_error(y_test, y_pred_custom)

print(f"\nKết quả CUSTOM:")
print(f"  R² Score: {r2_custom:.4f}")
print(f"  RMSE: {rmse_custom:.4f}")
print(f"  MAE: {mae_custom:.4f}")
print(f"  Non-zero coefficients: {np.sum(w != 0)}/{len(w)}")

# TRAIN SCIKIT-LEARN MODEL
print("\n2. SCIKIT-LEARN ELASTIC NET")
print("-" * 60)

from sklearn.linear_model import ElasticNet

# Tạo và train scikit-learn model
model_sklearn = ElasticNet(
    alpha=alpha,
    l1_ratio=l1_ratio,
    max_iter=1000,
    random_state=42,
    verbose=0  # Không in log
)

model_sklearn.fit(X_train_scaled, y_train)
y_pred_sklearn = model_sklearn.predict(X_test_scaled)

# Đánh giá scikit-learn model
r2_sklearn = r2_score(y_test, y_pred_sklearn)
rmse_sklearn = np.sqrt(mean_squared_error(y_test, y_pred_sklearn))
mae_sklearn = mean_absolute_error(y_test, y_pred_sklearn)

print(f"\nKết quả SCIKIT-LEARN:")
print(f"  R² Score: {r2_sklearn:.4f}")
print(f"  RMSE: {rmse_sklearn:.4f}")
print(f"  MAE: {mae_sklearn:.4f}")
print(f"  Non-zero coefficients: {np.sum(model_sklearn.coef_ != 0)}/{len(model_sklearn.coef_)}")

# COMPARISON
print("\n3. SO SÁNH")
print("-" * 60)
print(f"R² Score:")
print(f"  Custom:      {r2_custom:.4f} {'✓' if r2_custom > r2_sklearn else 'x'}")
print(f"  Scikit-learn: {r2_sklearn:.4f} {'✓' if r2_sklearn > r2_custom else 'x'}")
print(f"  Hiệu: {r2_sklearn - r2_custom:+.4f}")

print(f"\nRMSE (thấp hơn tốt hơn):")
print(f"  Custom:      {rmse_custom:.4f} {'✓' if rmse_custom < rmse_sklearn else 'x'}")
print(f"  Scikit-learn: {rmse_sklearn:.4f} {'✓' if rmse_sklearn < rmse_custom else 'x'}")
print(f"  Hiệu: {rmse_sklearn - rmse_custom:+.4f}")

print(f"\nMAE (thấp hơn tốt hơn):")
print(f"  Custom:      {mae_custom:.4f} {'✓' if mae_custom < mae_sklearn else 'x'}")
print(f"  Scikit-learn: {mae_sklearn:.4f} {'✓' if mae_sklearn < mae_custom else 'x'}")
print(f"  Hiệu: {mae_sklearn - mae_custom:+.4f}")

print("\n" + "=" * 60)
