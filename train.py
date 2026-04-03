import pandas as pd
import numpy as np
import pickle

from preprocess import clean_data, handle_outliers_iqr
from elasticnet import elastic_net, train_test_split, StandardScaler, predict, r2_score

# 1. Load & Clean
df = pd.read_csv("Data_CarPrice.csv")
df = clean_data(df)

numeric_cols = ['AskPrice', 'kmDriven', 'km_per_year']
df = handle_outliers_iqr(df, numeric_cols)

# 2. Split dữ liệu
X = df.drop(columns=['AskPrice'])
y = np.log1p(df['AskPrice'].values)

X_train_raw, X_test_raw, y_train, y_test = train_test_split(X.values, y, test_size=0.2)
# Chuyển lại về DataFrame để xử lý One-Hot Encoding
columns_raw = X.columns
X_train = pd.DataFrame(X_train_raw, columns=columns_raw)
X_test  = pd.DataFrame(X_test_raw, columns=columns_raw)

# 3. One-Hot Encoding 
categorical_features = ['Brand', 'model', 'Transmission', 'Owner', 'FuelType']

X_train = pd.get_dummies(X_train, columns=categorical_features, dtype=int)
X_test  = pd.get_dummies(X_test, columns=categorical_features, dtype=int)

# 4. Align columns 
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

# Lưu danh sách cột sau khi encoding
columns_final = X_train.columns.tolist()

# 5. Fix Dtype & Fillna
X_train = X_train.apply(pd.to_numeric, errors='coerce').fillna(0)
X_test  = X_test.apply(pd.to_numeric, errors='coerce').fillna(0)

# Chuyển sang Numpy
X_train_np = X_train.values.astype(np.float64)
X_test_np  = X_test.values.astype(np.float64)

# 6. Scale dữ liệu
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_np)
X_test_scaled  = scaler.transform(X_test_np)

# 7. Train Elastic Net
alpha = 0.00001
l1_ratio = 0.1

w, b = elastic_net(
    X_train_scaled, 
    y_train, 
    alpha=0.00001, 
    l1_ratio=0.1, 
    max_iter=2000, 
    tol=1e-7  
)

# 8. Save Model (Full Pipeline)
model = {
    "weights": w,
    "bias": b,
    "scaler": scaler,
    "columns": columns_final, 
    "alpha": alpha,
    "l1_ratio": l1_ratio
}

with open("elastic_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Đã lưu model vào elastic_model.pkl")

# 9. Evaluate
y_pred = predict(X_test_scaled, w, b)
y_test_real = np.expm1(y_test)
y_pred_real = np.expm1(y_pred)

print("R2 Score:", r2_score(y_test_real, y_pred_real))