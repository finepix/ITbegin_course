## 数据集
sklearn中digits数据集

## 任务

- 使用KNN分类算法对于其进行分类
- 使用流形学习算法对数据进行降维
- 观察降维之后的数据上的分类效果
- 试试不同的n_neighbor参数对于降维效果的影响
- 试试不同的维度，对于分类结果的影响

## 示例代码框架
```python
import numpy as np
import time

from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.manifold import LocallyLinearEmbedding


def load_dataset():
    """
            从sklearn加载数据
    :return:
    """
    return datasets.load_digits()

if __name__ == '__main__':

    # 加载数据，并对数据进行划分
    digits = load_dataset()

    X = digits.data
    Y = digits.target
    n_features = X.shape[1]
    n_classes = np.unique(Y).shape[0]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

    # TODO: 原始空间的分类效果

    # TODO: 使用lda对数据进行降维
    subspace_dim = 4

    # TODO: 在子空间上的分类效果
```


## 代码运行结果示例
> 由与数据划分的随机性，所以这里的结果仅仅作为参考。
```
原始空间的准确率：0.9870, 原始空间数据维度:64， 耗时：95 ms。
子空间的准确率：0.8574, 子空间数据维度：4， 耗时：3 ms。

原始空间的准确率：0.9833, 原始空间数据维度:64， 耗时：98 ms。
子空间的准确率：0.9870, 子空间数据维度：56， 耗时：48 ms。
```
从上面的结果可以得出结论：
- 大幅度降维会使得数据在一定程度上信息丢失；
- 适当的控制数据的维度，能在提高系统运行效率的同时增加算法的准确率。