## 数据集
使用sklearn自带的wine数据集（数据集分为三类）。
> 为了方便观察聚类中心，这里我们只取数据的前两列来训练
>
## 任务

- 对wine数据集进行kmeans聚类，并数据聚类中心，以及得到各种聚类的评价指标（互信息）


## 示例代码
```python
from sklearn import datasets, metrics
from sklearn.cluster import KMeans


def load_dataset():
    return datasets.load_wine()


if __name__ == '__main__':
    # 加载数据集
    wine = load_dataset()

    X = wine.data[:, :2]
    Y = wine.target
    # print(Y)

    # 初始化以及训练模型

    # 数据聚类结果以及聚类评价指标

```
## 示例运行结果

```
互信息：0.4457
聚类中心1：[13.0632  3.8948]
聚类中心2：[13.71538462  1.79969231]
聚类中心3：[12.21349206  1.6531746 ]
```