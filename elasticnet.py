import pandas as pd
import numpy as np

df = pd.read_csv('Data_Ready_For_ML.csv')

# Tách X và y
X = df.drop(columns=['AskPrice']).values.astype(float)
y = df['AskPrice'].values.astype(float)

print("Shape X:", X.shape)
print("Shape y:", y.shape)


# Chia train/test (80/20)
def train_test_split(X, y, test_size=0.2, random_state=42):
    np.random.seed(random_state)
    idx = np.random.permutation(len(y))
    split = int(len(y) * (1 - test_size))
    train_idx, test_idx = idx[:split], idx[split:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

# Chuẩn hoá
def standard_scaler(X_train, X_test):
    mean = X_train.mean(axis=0)
    std  = X_train.std(axis=0)
    std[std == 0] = 1  # tránh chia 0
    return (X_train - mean) / std, (X_test - mean) / std

X_train, X_test, y_train, y_test = train_test_split(X, y)
X_train_s, X_test_s = standard_scaler(X_train, X_test)

print("Train:", X_train.shape)
print("Test :", X_test.shape)

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

    for iteration in range(max_iter):
        w_old = w.copy()

        # Cập nhật bias (không bị phạt)
        residual = y - X @ w - b
        b = b + residual.mean()

        # Cập nhật từng hệ số w_j (Coordinate Descent)
        for j in range(p):
            # Tính residual không tính đóng góp của w_j
            residual = y - X @ w - b + X[:, j] * w[j]

            # Tính rho: tương quan giữa feature j và residual
            rho = (X[:, j] @ residual) / n

            # Cập nhật w_j bằng soft threshold + penalty L2
            w[j] = soft_threshold(rho, alpha_l1) / (1 + alpha_l2)

        # Kiểm tra hội tụ
        if np.max(np.abs(w - w_old)) < tol:
            print(f"  Hội tụ sau {iteration+1} vòng lặp")
            break

    return w, b