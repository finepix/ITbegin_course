## 数据集
使用sklearn自带的boston房价数据集

## 任务
使用svr对Boston房价进行预测（由于是回归任务，所以这里采用MSE和MAE作为评价标准）

## 示例代码部分

```python 
from sklearn import datasets, svm
from sklearn.model_selection import train_test_split
from sklearn import metrics


def load_dataset():
    """
            加载数据
    :return:
    """
    return datasets.load_boston()


if __name__ == '__main__':
    boston = load_dataset()

    # 划分训练测试集
    X = boston.data
    Y = boston.target
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

    # step1: 训练

    # step2: 测试

    # 打印分类结果

    # 计算准确度等度量
```

## 运行示例结果
> 由于数据划分的随机性，所以结果不尽相同，大家可以尝试多种参数，最好可以做交叉验证来评估模型的好坏。

```
绝对误差：4.8648
均方误差：51.6949
```