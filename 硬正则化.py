import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 设置随机种子
np.random.seed(123)

# 创建数据
n_dots = 20
x = np.linspace(0, 1, n_dots)
y = np.sqrt(x) + 0.2 * np.random.rand(n_dots) - 0.1

# ===============================
# 硬正则化：只使用前 4 个多项式特征
# ===============================
degree_limit = 4  # 使用 1~4 阶特征（硬约束）
x_l = [np.power(x, i + 1).reshape(-1, 1) for i in range(degree_limit)]
X = np.concatenate(x_l, axis=1)

# 测试集构建
t = np.linspace(0, 1, 200)
t_l = [np.power(t, i + 1).reshape(-1, 1) for i in range(degree_limit)]
T = np.concatenate(t_l, axis=1)

# 模型训练（线性回归）
lr_hard = LinearRegression()
lr_hard.fit(X, y)

# ===============================
# 与“未正则化”全10阶模型对比
# ===============================
x_full = [np.power(x, i + 1).reshape(-1, 1) for i in range(10)]
X_full = np.concatenate(x_full, axis=1)
t_full = [np.power(t, i + 1).reshape(-1, 1) for i in range(10)]
T_full = np.concatenate(t_full, axis=1)

lr_full = LinearRegression()
lr_full.fit(X_full, y)

# 可视化结果
plt.figure(figsize=(10, 4))

# 左图：硬正则化（只用4阶）
plt.subplot(1, 2, 1)
plt.plot(x, y, 'ro', label='Data')
plt.plot(t, lr_hard.predict(T), 'b-', label='Hard Regularization (4-degree)')
plt.plot(t, np.sqrt(t), 'r--', label='True Function')
plt.title('Hard Regularization')
plt.legend()

# 右图：10阶全特征拟合（容易过拟合）
plt.subplot(1, 2, 2)
plt.plot(x, y, 'ro', label='Data')
plt.plot(t, lr_full.predict(T_full), 'g-', label='Full 10-degree Fit')
plt.plot(t, np.sqrt(t), 'r--', label='True Function')
plt.title('Full 10-degree Polynomial')
plt.legend()

plt.tight_layout()
plt.show()
