## 数据集
使用sklearn自带的wine数据集（数据集分为三类）。

## 任务


- 使用sklearn的谱聚类对wine数据集进行聚类；
- 并将得到的聚类结果互信息与kmeans算法得到的结果进行对比分析；
- 调参，试试多个参数对于聚类结果的影响

## 示例代码
```python
import numpy as np

from sklearn import datasets, metrics
from sklearn.cluster import SpectralClustering

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
互信息：0.3827
模型：SpectralClustering(gamma=0.2, n_clusters=3, random_state=4399)
```