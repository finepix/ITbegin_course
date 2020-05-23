from __future__ import print_function

from time import time
import logging
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.svm import SVC

print(__doc__)  # 输出文件开头注释的内容

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

# 从外网下载数据集，若数据集已经在本地磁盘就直接导入，这里建议若没有vpn先下载好按照教程中给的链接地址放好数据，直接调用该方法即可
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

# 图像矩阵的行h,列w
n_samples, h, w = lfw_people.images.shape
# 图片数据
X = lfw_people.data
# 矩阵列数特征点数据1850
n_features = X.shape[1]

# y是label,有7个目标时，0-6之间取值
y = lfw_people.target

# 字符串，表示人名
target_names = lfw_people.target_names
# shape[0]--行维数 shape[1]--列维数
n_classes = target_names.shape[0]
print("Total dataset size:")
print("n_samples: %d\nn_features: %d\nn_classes: %d" % (n_samples, n_features, n_classes))

# 分离测试集和训练集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25)
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

# 由于数据维度较大，这里先做一个pca
n_components = 150
print("Extracting the top %d eigen faces from %d faces" % (n_components, X_train.shape[0]))
t0 = time()
pca = PCA(svd_solver='randomized', n_components=n_components, whiten=True)
# pca训练
pca.fit(X, y)
print("done in %0.3fs" % (time() - t0))
eigenfaces = pca.components_.reshape((n_components, h, w))
print("Projecting the input data on the eigenfaces orthonormal basis")
t0 = time()
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
print("done in %0.3fs" % (time() - t0))

# 训练SVM分类器
print("Fitting the classifier to the training set")
t0 = time()
# 网格搜索法搜索参数
param_grid = {'C': [1e3, 998, 1001, 999, 1002],
              'gamma': [0.0001, 0.003, 0.0035, 0.004, 0.0045], }
# clf = GridSearchCV(SVC(kernel='rbf', class_weight='auto'), param_grid)
clf = GridSearchCV(SVC(kernel='rbf', class_weight=None), param_grid)
# 分类器训练
clf = clf.fit(X_train_pca, y_train)
print("done in %0.3fs" % (time() - t0))
print("Best estimator found by grid search:")
# 训练得到的最优参数
print(clf.best_estimator_)
# 得到预测结果
print("Predicting people's names on the test set")
t0 = time()
y_pred = clf.predict(X_test_pca)
print("done in %0.3fs" % (time() - t0))
# 对预测结果进行评价
print(classification_report(y_test, y_pred, target_names=target_names))
print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))


def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    """
    Helper function to plot a gallery of portraits
    :param images:
    :param titles:
    :param h:
    :param w:
    :param n_row:
    :param n_col:
    :return:
    """
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())


def title(y_pred, y_test, target_names, i):
    """
    plot the result of the prediction on a portion of the test set

    :param y_pred:
    :param y_test:
    :param target_names:
    :param i:
    :return:
    """
    pred_name = target_names[y_pred[i]].rsplit(' ', 1)[-1]
    true_name = target_names[y_test[i]].rsplit(' ', 1)[-1]
    return 'predicted: %s\ntrue:      %s' % (pred_name, true_name)


# 画出预测结果
prediction_titles = [title(y_pred, y_test, target_names, i)
                     for i in range(y_pred.shape[0])]
plot_gallery(X_test, prediction_titles, h, w)

# 画出特征脸
eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
plot_gallery(eigenfaces, eigenface_titles, h, w)

plt.show()
