import numpy as np
import matplotlib.pyplot as plt
import sys


def main(args):
    a = np.array([1010, 1000, 990])
    print(softmax(a))

    plotFunction(F(0.5, 0.5, -0.7))
    plogFunc(step_function)
    plogFunc(sigmoid)


def plogFunc(func):
    x = np.arange(-5.0, 5.0, 0.1)
    y = func(x)

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.plot(x, y)
    ax1.set_ylim(-0.1, 1.1)
    # ax1.set_aspect('equal')

    plt.show()


def F(w1, w2, b):
    _x_set = np.arange(-5.0, 5.0, 0.1)

    # _x_set = np.linspace(-1.5, 1.5, 50)

    def _f(x):
        return -(w1 * x + b) / w2

    return _x_set, _f


def plotFunction(X):
    x, f = X
    y = f(x)

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)

    ax1.plot(x, y)  # 直線を引く
    ax1.set_aspect('equal')

    ax1.grid(True, which='both')
    ax1.axhline(y=0, color='k')
    ax1.axvline(x=0, color='k')
    ax1.set_xlim([-2, 2])
    ax1.set_ylim([-2, 2])

    plt.show()  # グラフ表示

def mean_squared_error(y, t):
    return 0.5 * np.sum((y - t) ** 2)


def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)


    batch_size = y.shape[0]
    delta = 1e-7
    return -np.sum(t * np.log(y + delta)) / batch_size


def step_function(x):
    """
    やってることは
    if x <= 0:
      return 0
    elif x > 0:
      return 1
    だけど、np.arrayを引数に取れるバージョン
    :param x:
    :return:
    """
    return np.array(x > 0, dtype=np.int)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y


def perceptron(x1, x2, w1, w2, b):
    w = np.array([w1, w2])
    x = np.array([x1, x2])
    tmp = np.sum(w * x) + b
    return step_function(tmp)


def AND(x1, x2):
    # (1,1)(x1,x2) - 1.4 = 0 な直線
    return perceptron(x1, x2, w1=0.5, w2=0.5, b=-0.7)


def NAND(x1, x2):
    return perceptron(x1, x2, w1=-0.5, w2=-0.5, b=0.7)


def OR(x1, x2):
    return perceptron(x1, x2, w1=0.5, w2=0.5, b=-0.2)


def XOR(x1, x2):
    return AND(NAND(x1, x2), OR(x1, x2))


if __name__ == "__main__":
    main(sys.argv)
