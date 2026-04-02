import pandas as pd
import numpy as np
import pickle

from preprocess import clean_data, handle_outliers_iqr
from encoding import fit_target_encoder, transform_target_encoder
from elasticnet import elastic_net, train_test_split, StandardScaler, predict, r2_score

# Load & Clean
df = pd.read_csv("Data_CarPrice.csv")
df = clean_data(df)

numeric_cols = ['AskPrice', 'kmDriven', 'km_per_year']
df = handle_outliers_iqr(df, numeric_cols)

# Split BEFORE encoding
X = df.drop(columns=['AskPrice'])
y = df['AskPrice']

X_train_np, X_test_np, y_train, y_test = train_test_split(X.values, y.values)

# Convert lại DataFrame
columns_raw = X.columns
X_train = pd.DataFrame(X_train_np, columns=columns_raw)
X_test  = pd.DataFrame(X_test_np, columns=columns_raw)

# Target Encoding (NO LEAK)
global_mean = y_train.mean()
target_maps = {}

for col in ['Brand', 'model']:
    if col in X_train.columns:
        train_df = X_train.copy()
        train_df['AskPrice'] = y_train

        mean_map = fit_target_encoder(train_df, col, 'AskPrice')
        target_maps[col] = mean_map  

        X_train[col] = transform_target_encoder(X_train, col, mean_map, global_mean)
        X_test[col]  = transform_target_encoder(X_test, col, mean_map, global_mean)

# One-hot encoding
X_train = pd.get_dummies(X_train, drop_first=True)
X_test  = pd.get_dummies(X_test, drop_first=True)

# Align columns (QUAN TRỌNG NHẤT)
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

# Lưu columns SAU encoding (đúng chỗ)
columns_final = X_train.columns.tolist()

# FIX DTYPE
X_train = X_train.apply(pd.to_numeric, errors='coerce')
X_test  = X_test.apply(pd.to_numeric, errors='coerce')

X_train = X_train.fillna(0)
X_test  = X_test.fillna(0)

# Convert sang numpy
X_train = X_train.values.astype(np.float64)
X_test  = X_test.values.astype(np.float64)

# Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# Train Elastic Net
alpha = 0.1
l1_ratio = 0.5

w, b = elastic_net(
    X_train,
    y_train,
    alpha=alpha,
    l1_ratio=l1_ratio,
    max_iter=1000
)

# Save Model (FULL PIPELINE)
model = {
    "weights": w,
    "bias": b,
    "scaler": scaler,
    "columns": columns_final,     
    "target_maps": target_maps, 
    "global_mean": global_mean,
    "alpha": alpha,
    "l1_ratio": l1_ratio
}

with open("elastic_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Đã lưu model vào elastic_model.pkl")

# Evaluate
y_pred = predict(X_test, w, b)

print("R2 Score:", r2_score(y_test, y_pred))