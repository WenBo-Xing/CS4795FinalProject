import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
import pandas as pd

# 指定数据集下载和缓存的路径
data_home = '/Users/xingwenbo/Documents/CS4795'
mnist = fetch_openml('mnist_784', data_home=data_home, version=1, as_frame=False)

X, y = mnist["data"], mnist["target"]

# 选择要显示的样本数量
num_samples_to_display = 5

for i in range(num_samples_to_display):
    # 将一维数组重塑为28x28的二维数组
    image = X[i].reshape(28, 28)

    # 使用matplotlib显示图像
    plt.imshow(image, cmap='gray')
    plt.title(f"Label: {y[i]}")
    plt.show()