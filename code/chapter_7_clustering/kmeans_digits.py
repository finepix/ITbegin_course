from time import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale


def plot_digits(_data, col=3, row=8, image_size=(8, 8)):
    """
            绘制部分手写数字
    :param _data:
    :param col: 列数
    :param row: 行数
    :param image_size: 图像大小，数字集的大小为8*8
    :return:
    """
    fig, ax = plt.subplots(nrows=row, ncols=col)
    ax = ax.flatten()
    for i in range(row * col):
        tmp_data = _data[i, :].reshape(image_size)
        ax[i].imshow(tmp_data, cmap='Greys', interpolation='nearest')
        ax[i].set_xticks(())
        ax[i].set_yticks(())
    fig.subplots_adjust(left=0.02, bottom=0.02, right=0.98, top=0.98, wspace=0.002, hspace=0.002)
    fig.show()


np.random.seed(42)

# 导入手写数字集
X_digits, y_digits = load_digits(return_X_y=True)
# 归一化
data = scale(X_digits)

n_samples, n_features = data.shape
n_digits = len(np.unique(y_digits))
labels = y_digits

# 类别数：10， 样本数：1797， 特征维度：64
print("n_digits: %d, \t n_samples %d, \t n_features %d" % (n_digits, n_samples, n_features))

# 部分数字可视化
h_samples = 8
v_samples = 3
plot_data = data[:h_samples * v_samples, :]
plot_digits(plot_data, row=v_samples, col=h_samples, image_size=(8, 8))

# 数据可视化（将原始的数据使用pca降维到2维之后绘制聚类结果）
reduced_data = PCA(n_components=2).fit_transform(data)

# k-means算法
k_means = KMeans(init='k-means++', n_clusters=n_digits, n_init=10)
k_means.fit(reduced_data)

# 控制聚类边缘绘图的步长
h = .02

# 绘制出决策边界，并为每一个区域分配不同的颜色
x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# 使用k-means得到每个区域的颜色
Z = k_means.predict(np.c_[xx.ravel(), yy.ravel()])

Z = Z.reshape(xx.shape)
plt.figure(1)
plt.clf()
# 决策边界
plt.imshow(Z, interpolation='nearest', extent=(xx.min(), xx.max(), yy.min(), yy.max()), cmap=plt.cm.Paired,
           aspect='auto', origin='lower')

# 绘制数据点
plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)

# 绘制簇中心点（使用白色X）
centroids = k_means.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=169, linewidths=3, color='w', zorder=10)

plt.title('K-means clustering on the digits dataset')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())

plt.show()
