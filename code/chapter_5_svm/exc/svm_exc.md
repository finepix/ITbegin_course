## 数据集
使用sklearn自带的iris数据集

## 任务
使用svm对其进行分类，并输出分类结果，计算分类指标

## 示例代码部分

```python 
from sklearn import datasets

def load_dataset():
    """
            加载数据
    :return:
    """
    return datasets.load_iris()


if __name__ == '__main__':
    iris = load_dataset()

    # 划分训练测试集
    X = iris.data
    Y = iris.target
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

    # 使用svm分类器分类

    # 打印分类结果

    # 计算准确度等度量

```

## 运行结果如下

```
预测结果：[1 2 1 1 2 0 1 0 0 0 2 1 2 2 1 2 1 1 2 1 1 2 0 1 0 0 2 0 1 2 0 1 0 1 2 0 0
 1 0 2 1 1 1 2 1]
测试准确率：0.9778
```