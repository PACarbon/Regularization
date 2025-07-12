import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso

# 设置随机种子
np.random.seed(123)

# 创建数据
n_dots = 20
x = np.linspace(0, 1, n_dots)
y = np.sqrt(x) + 0.2 * np.random.rand(n_dots) - 0.1

# 多项式特征（10阶）
x_l = []
for i in range(10):
    x_temp = np.power(x, i + 1).reshape(-1, 1)
    x_l.append(x_temp)
X = np.concatenate(x_l, axis=1)

# 普通线性回归
lr = LinearRegression()
lr.fit(X, y)

# 岭回归
reg_rid = Ridge(alpha=0.005)
reg_rid.fit(X, y)

# 可视化结果
t = np.linspace(0, 1, 200)
t_l = [np.power(t, i + 1).reshape(-1, 1) for i in range(10)]
T = np.concatenate(t_l, axis=1)

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(x, y, 'ro', label='Data')
plt.plot(t, reg_rid.predict(T), 'b-', label='Ridge')
plt.plot(t, np.sqrt(t), 'r--', label='True Function')
plt.title('Ridge(alpha=0.005)')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(x, y, 'ro', label='Data')
plt.plot(t, lr.predict(T), 'g-', label='Linear')
plt.plot(t, np.sqrt(t), 'r--', label='True Function')
plt.title('Linear Regression')
plt.legend()

plt.tight_layout()
plt.show()
