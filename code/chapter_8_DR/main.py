import matplotlib.pyplot as plt
import numpy as np

from scipy.io import loadmat
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import LocallyLinearEmbedding, TSNE

# 手写数字集文件地址（请下载好数据之后放好文件位置，绝对地址或者相对地址，基础知识）
FILE_PATH_USPS_DATASET = '../data/USPS.mat'


def load_usps_from_file(file_path=FILE_PATH_USPS_DATASET, n_samples=1000, reshuffle=True):
    """
            从文件中导入USPS手写数字集
    :param file_path: 文件地址
    :param n_samples: 最后实验需要的样本数量
    :param reshuffle: 是否打乱数据集顺序， true表示打乱顺序然后再截取部分数据
    :return:
    """
    # 读取数据，fea对应其特征，gnd为标签
    _data = loadmat(file_path)
    _fea = _data['fea']
    _gnd = _data['gnd']

    total_samples = _data['fea'].shape[0]

    # 判断实验所需样本是否小于总的样本数
    assert n_samples < total_samples

    # 打乱数据集顺序
    if reshuffle:
        per_index = np.random.permutation(total_samples)
        _fea = _fea[per_index, :]
        _gnd = _gnd[per_index]

    # 截取前n的样本
    _fea = _fea[:n_samples, :]
    _gnd = _gnd[: n_samples]

    return _fea, _gnd


def visual_on_origin_data(fea, margin=0, img_size=(16, 16), n_img_row=20, n_img_col=28):
    """
            原始usps数据的样本
    :param fea: usps的特征n*256，每一个样本是原始图像拉伸得到的256*1的向量
    :param margin: 间隔
    :param img_size: 原始图像的尺寸
    :param n_img_row: 每行的图像个数
    :param n_img_col: 每列的图像个数
    :return:
    """
    img_width = img_size[1]
    width = img_width + margin

    img = np.zeros((width * n_img_row, width * n_img_col))
    for i in range(n_img_row):
        ix = width * i
        for j in range(n_img_col):
            iy = width * j
            img[ix:ix + img_width, iy:iy + img_width] = fea[i * n_img_row + j].reshape(img_size)

    plt.imshow(img, cmap=plt.cm.binary)
    plt.xticks([])
    plt.yticks([])
    plt.title('Visualization on USPS')
    plt.show()


def plot_embedding(fea, label, title=None):
    """
            绘制降维后的图像
    :param fea: 特征n*2
    :param label: 特征标签
    :param title: 图像的标题
    :return:
    """
    # 数据规范化
    fea_min, fea_max = np.min(fea, 0), np.max(fea, 0)
    fea = (fea - fea_min) / (fea_max - fea_min)

    plt.figure()
    for i in range(fea.shape[0]):
        label_i = label[i, 0]
        # 将数字标记在该坐标
        plt.text(fea[i, 0], fea[i, 1], str(label_i), color=plt.cm.Set1(label_i / 10.),
                 fontdict={'weight': 'bold', 'size': 9})

    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)
    plt.show()


#############################################################################

# 拿到数据，特征、标签
usps_fea, usps_gnd = load_usps_from_file(FILE_PATH_USPS_DATASET)

# 原始样本
visual_on_origin_data(usps_fea)

# PCA
pca = PCA(n_components=2)
new_fea_pca = pca.fit_transform(usps_fea)
# 降维后的样本可视化
pca_title = 'DR on PCA'
plot_embedding(new_fea_pca, usps_gnd, title=pca_title)

# LDA
lda = LinearDiscriminantAnalysis(n_components=2)
new_fea_lda = lda.fit_transform(usps_fea, usps_gnd)
# 降维后的样本可视化
lda_title = 'DR on LDA'
plot_embedding(new_fea_lda, usps_gnd, title=lda_title)

# LLE
lle = LocallyLinearEmbedding(n_neighbors=5, n_components=2, method='standard')
new_fea_lle = lle.fit_transform(usps_fea)
# 降维后的样本可视化
lle_title = 'DR on LLE'
plot_embedding(new_fea_lle, usps_gnd, title=lle_title)

# T-SNE
t_sne = TSNE(n_components=2, init='pca')
new_fea_t_sne = t_sne.fit_transform(usps_fea)
# 降维后的样本可视化
t_sne_title = 'DR on t-sne'
plot_embedding(new_fea_t_sne, usps_gnd, title=t_sne_title)
