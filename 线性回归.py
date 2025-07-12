import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 设计随机数种子
np.random.seed(123)

# 创建数据
n_dots = 20
x = np.linspace(0, 1, n_dots)                        # 从0到1，等宽排布的20个数
y = np.sqrt(x) + 0.2*np.random.rand(n_dots) - 0.1    # 带有随机噪声的目标值  根号x是目标函数的真实值   0.2*np.random.rand(n_dots) - 0.1加入随机噪声

# 创建多项式特征
x_l = []
for i in range(10):
    x_temp = np.power(x, i+1).reshape(-1, 1)      #np.power(x, i+1)将 x 分别提升到 1~10 次幂，生成多项式特征  .reshape(-1, 1)：将结果转为二维列向量
    x_l.append(x_temp)

# 合并特征
X = np.concatenate(x_l, 1)  

# 拟合线性回归模型
lr = LinearRegression()
lr.fit(X, y)

# 观察建模结果
t = np.linspace(0, 1, 200)
plt.plot(x, y, 'ro', label='Data')
plt.plot(x, lr.predict(X), '-', label='Model Prediction')
plt.plot(t, np.sqrt(t), 'r--', label='True Function')
plt.title('10-degree Polynomial Regression')
plt.legend()
plt.show()
