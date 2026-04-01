import numpy as np
import pandas as pd
from elasticnet import elastic_net, standard_scaler, train_test_split # Lấy các hàm từ file elasticnet.py của nhóm

# 1. Load dữ liệu đã được encoding
df = pd.read_csv('Data_Ready_For_ML.csv')
X = df.drop(columns=['AskPrice']).values.astype(float)
y = df['AskPrice'].values.astype(float)

# 2. Chia tập dữ liệu và Chuẩn hóa (Standardization)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_s, X_test_s = standard_scaler(X_train, X_test)

# 3. Hàm tính RMSE (Root Mean Squared Error)
def calculate_rmse(y_true, y_pred):
    mse = np.mean((y_true - y_pred) ** 2)
    return np.sqrt(mse)

# 4. TÌM THAM SỐ TỐI ƯU (Tối ưu hóa Alpha và L1_Ratio)
alphas = [0.01, 0.1, 1.0, 10.0]
l1_ratios = [0.2, 0.5, 0.8]

best_rmse = float('inf')
best_params = {'alpha': None, 'l1_ratio': None}

print("Đang quét tìm tham số tối ưu...")

for a in alphas:
    for l1 in l1_ratios:
        # Huấn luyện mô hình
        w, b = elastic_net(X_train_s, y_train, alpha=a, l1_ratio=l1)
        
        # Dự đoán trên tập Test
        y_pred = X_test_s @ w + b
        
        # Tính sai số
        current_rmse = calculate_rmse(y_test, y_pred)
        
        # Cập nhật tham số tốt nhất
        if current_rmse < best_rmse:
            best_rmse = current_rmse
            best_params['alpha'] = a
            best_params['l1_ratio'] = l1

# 5. KẾT QUẢ CUỐI CÙNG
print("-" * 30)
print(f"Tham số tối ưu tìm được:")
print(f" - Alpha: {best_params['alpha']}")
print(f" - L1 Ratio: {best_params['l1_ratio']}")
print(f"Chỉ số RMSE thấp nhất (hệ log): {best_rmse:.4f}")

# Chạy lại mô hình MỘT LẦN NỮA với tham số TỐT NHẤT để lấy đúng w, b
w_best, b_best = elastic_net(X_train_s, y_train, alpha=best_params['alpha'], l1_ratio=best_params['l1_ratio'])
y_pred_best = X_test_s @ w_best + b_best

# Tính lại RMSE theo giá tiền thật
real_rmse = calculate_rmse(np.expm1(y_test), np.expm1(y_pred_best))
print(f"RMSE theo giá trị tiền thực tế: {real_rmse:,.0f} VND")
print("-" * 30)