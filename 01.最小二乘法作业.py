import numpy as np
from scipy.optimize import leastsq
import matplotlib.pyplot as plt

def fun(arr, x):
    """
    拟合函数
    :param arr: w斜率，b截距
    :param x: 自变量
    :return: y估计值
    """
    w, b = arr
    return w * x + b


def error(arr, x, y):
    return fun(arr, x) - y


def main():
    X_train = np.array([162, 165, 159, 173, 157, 175, 161, 164, 172, 158])
    Y_train = np.array([48, 64, 53, 66, 52, 68, 50, 52, 64, 49])
    x = np.linspace(150, 180, 1000)
    p = [1, 2]
    para = leastsq(error, p, args=(X_train, Y_train))
    y_fitted = fun(para[0], x)

    plt.figure()
    plt.scatter(X_train, Y_train, color='red', label='Sample Point')
    plt.plot(x, y_fitted, color='blue', label='Fitted curve')
    plt.legend()
    plt.show()
    print(para[0])

if __name__ == '__main__':
    main()
