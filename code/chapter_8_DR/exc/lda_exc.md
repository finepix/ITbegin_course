## 数据集
sklearn中digits数据集

## 任务

- 使用KNN分类算法对于其进行分类
- 使用lda对于数据降维
- 观察降维之后的数据上的分类效果
- 注意有监督降维和无监督降维算法的区别

## 示例代码框架
```python

from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


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
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

    # TODO: 原始空间的分类效果
    clf = KNeighborsClassifier(n_neighbors=5, weights='distance')
    clf.fit(X_train, Y_train)
    prediction = clf.predict(X_test)

    acc_origin_space = metrics.accuracy_score(Y_test, prediction)
    print('原始空间的准确率：%.4f, 原始空间数据维度:%d' % (acc_origin_space, n_features))

    # TODO: 使用LDA对数据进行降维

    # TODO: 在子空间上的分类效果
```


## 代码运行结果示例
> 由与数据划分的随机性，所以这里的结果仅仅作为参考。
```
原始空间的准确率：0.9852, 原始空间数据维度:64， 耗时：98 ms。
子空间的准确率：0.9648, 子空间数据维度：9， 耗时：10 ms。
```