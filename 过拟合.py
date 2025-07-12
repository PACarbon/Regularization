import numpy as np
import matplotlib.pyplot as plt

# 设定随机数种子
np.random.seed(123)

# 创建数据
n_dots = 20
x = np.linspace(0, 1, n_dots)  # 从0到1，等宽排布的20个数
y = np.sqrt(x) + 0.2*np.random.rand(n_dots) - 0.1  # 基于x生成y值，并添加一些噪声

def plot_polynomial_fit(x, y, deg):
    """
    对给定数据进行多项式拟合，并绘制出原始数据、拟合结果和理想结果的图像。
    
    参数
    ----------
    x : ndarray
        x坐标的数据
    y : ndarray
        y坐标的数据
    deg : int
        多项式的阶数
    """
    # 对数据进行多项式拟合
    p = np.poly1d(np.polyfit(x, y, deg))
    
    # 生成用于绘图的数据
    t = np.linspace(0, 1, 200)
    
    # 绘制原始数据（红色圆点）、拟合结果（蓝色实线）和理想结果（红色虚线）
    plt.plot(x, y, 'ro', label='Original Data')
    plt.plot(t, p(t), '-', label=f'Degree {deg} Fit')
    plt.plot(t, np.sqrt(t), 'r--', label='Ideal Result')
    
    # 显示图例
    plt.legend()

plt.figure(figsize=(18, 4), dpi=200)
degrees = [1, 3, 10]  # 多项式的阶数
titles = ['Under Fitting', 'Fitting', 'Over Fitting']  # 图像的标题
for index, deg in enumerate(degrees):
    plt.subplot(1, 3, index + 1)
    plot_polynomial_fit(x, y, deg)
    plt.title(titles[index], fontsize=20)

plt.show()


