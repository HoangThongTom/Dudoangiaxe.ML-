import pandas as pd
import numpy as np

def predict(X, w, b):
    return X @ w + b

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - y_true.mean())**2)
    return 1 - ss_res / ss_tot

def compute_loss(X, y, w, b, alpha, l1_ratio):
    n = len(y)
    y_pred = X @ w + b

    mse = np.mean((y - y_pred)**2)
    l1 = np.sum(np.abs(w))
    l2 = np.sum(w**2)

    return mse + alpha * (l1_ratio * l1 + 0.5 * (1 - l1_ratio) * l2)

# Chia train/test (80/20)
def train_test_split(X, y, test_size=0.2, random_state=42):
    np.random.seed(random_state)
    idx = np.random.permutation(len(y))
    split = int(len(y) * (1 - test_size))
    train_idx, test_idx = idx[:split], idx[split:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

# Chuẩn hoá
class StandardScaler:
    def fit(self, X):
        X = np.array(X, dtype=np.float64)   # ✅ ép kiểu cứng
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)
        self.std[self.std == 0] = 1

    def transform(self, X):
        X = np.array(X, dtype=np.float64)   # ✅ đảm bảo luôn đúng
        return (X - self.mean) / self.std

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

def soft_threshold(rho, alpha_l1):
    """
    Hàm co rút (shrinkage) cho penalty L1
    - Nếu rho > alpha_l1  : trả về rho - alpha_l1  (dương)
    - Nếu rho < -alpha_l1 : trả về rho + alpha_l1  (âm)
    - Nếu ở giữa         : trả về 0 (loại biến luôn)
    """
    if rho > alpha_l1:
        return rho - alpha_l1
    elif rho < -alpha_l1:
        return rho + alpha_l1
    else:
        return 0.0

def elastic_net(X, y, alpha=1.0, l1_ratio=0.5, max_iter=1000, tol=1e-4):
    """
    Elastic Net bằng Coordinate Descent

    Hàm loss:
    L = (1/n) * sum(y - Xw - b)^2
      + alpha * l1_ratio * sum|w|          ← penalty L1 (Lasso)
      + 0.5 * alpha * (1-l1_ratio) * sum(w^2)  ← penalty L2 (Ridge)

    Tham số:
    - alpha    : mức độ phạt tổng thể
    - l1_ratio : tỷ lệ L1 (0=Ridge, 1=Lasso, 0.5=cân bằng)
    - max_iter : số vòng lặp tối đa
    - tol      : ngưỡng hội tụ
    """
    n, p = X.shape
    w = np.zeros(p)  # khởi tạo tất cả hệ số = 0
    b = 0.0          # bias

    alpha_l1 = alpha * l1_ratio          # phần L1
    alpha_l2 = alpha * (1 - l1_ratio)    # phần L2

# Giải pháp tối ưu cho hàm elastic_net
    for iteration in range(max_iter):
        w_old = w.copy()
        
        # 1. Cập nhật bias (b) một lần cho mỗi vòng lặp lớn
        y_pred = X @ w + b
        b = b + np.mean(y - y_pred)
        
        # 2. Cập nhật từng w_j
        y_pred = X @ w + b # Cập nhật lại y_pred sau khi đổi b
        for j in range(p):
            # Tính rho mà không cần nhân lại toàn bộ ma trận
            # residual_j = y - (y_pred - X[:, j] * w[j])
            rho = (X[:, j] @ (y - y_pred + X[:, j] * w[j])) / n
            
            z_j = (X[:, j] @ X[:, j]) / n
            new_w_j = soft_threshold(rho, alpha_l1) / (z_j + alpha_l2)
            
            # Cập nhật y_pred ngay lập tức để j tiếp theo sử dụng thông tin mới nhất
            y_pred = y_pred + X[:, j] * (new_w_j - w[j])
            w[j] = new_w_j
        # Kiểm tra hội tụ
        print(f"  Loop: {iteration+1} - Loss: {compute_loss(X, y, w, b, alpha, l1_ratio):.4f}")
        if np.max(np.abs(w - w_old)) < tol:
            print(f"  Hội tụ sau {iteration+1} vòng lặp")
            break

    return w, b

