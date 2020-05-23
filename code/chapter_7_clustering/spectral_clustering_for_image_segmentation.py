import time

import numpy as np
from distutils.version import LooseVersion
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt
import skimage
from skimage.data import coins
from skimage.transform import rescale

from sklearn.feature_extraction import image
from sklearn.cluster import spectral_clustering


def show_ori_img(img_data, text='origin image'):
    """
            将原始图像绘制出来
    :param img_data:
    :param text: 图像title
    :return:
    """
    plt.figure()

    plt.imshow(img_data)

    plt.xticks(())
    plt.yticks(())
    plt.title(text)
    plt.show()


def show_result_img(img_data, result_label, t, options):
    """
            绘制分割图像
    :param img_data: 原始图像数据
    :param result_label: 谱聚类得到的label
    :param t: 时间
    :param options: 用于标签分配的方法（k-means等）
    :return:
    """
    global N_REGIONS

    plt.figure(figsize=(5, 5))
    plt.imshow(img_data, cmap=plt.cm.gray)

    # 分割硬币
    for la in range(N_REGIONS):
        plt.contour(result_label == la, colors=[plt.cm.nipy_spectral(la / float(N_REGIONS))])

    plt.xticks(())
    plt.yticks(())
    text = 'Spectral clustering: %s, %.2fs' % (options, t)
    print(text)
    plt.title(text)
    plt.show()


# 判断版本
if LooseVersion(skimage.__version__) >= '0.14':
    rescale_params = {'anti_aliasing': False, 'multichannel': False}
else:
    rescale_params = {}

# 导入原始的数据
orig_coins = coins()

# 使用高斯滤波来平滑区域达到降采样减少锯齿伪影
smooth_coins = gaussian_filter(orig_coins, sigma=2)
# 将图像缩小一定的比例，加速处理的过程
rescaled_coins = rescale(smooth_coins, 0.2, mode="reflect", **rescale_params)

# 将处理过后的图像画出来
show_ori_img(rescaled_coins, 'origin coin image')

# 将图像转化为图（将图像的梯度作为边的权重）
graph = image.img_to_graph(rescaled_coins)

beta = 10
eps = 1e-6
graph.data = np.exp(-beta * graph.data / graph.data.std()) + eps

# 进行谱聚类
# 这里设置25个区域是有一定道理的，由于有24个硬币再加上背景，所以将图分为25个区域
N_REGIONS = 25
#############################################################################

# 可视化最终的分割结果，这里分为两种方式
for option in ('kmeans', 'discretize'):
    t0 = time.time()
    # 谱聚类步骤
    labels = spectral_clustering(graph, n_clusters=N_REGIONS, assign_labels=option, random_state=42)
    t1 = time.time()
    labels = labels.reshape(rescaled_coins.shape)
    # 绘制结果图
    show_result_img(rescaled_coins, labels, (t1 - t0), option)

