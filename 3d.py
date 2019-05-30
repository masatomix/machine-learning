import sys
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


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


def h(x):
    return step_function(x)
    # return sigmoid(x)


def XOR_3d(X, Y):
    n = NAND(X, Y)
    n = h(n)
    o = OR(X, Y)
    o = h(o)

    a = AND(n, o)
    a = h(a)

    return a


def perceptron(x1, x2, w1, w2, b):
    return w1 * x1 + w2 * x2 + b


def AND(x1, x2):
    # (1,1)(x1,x2) - 1.4 = 0 な直線
    return perceptron(x1, x2, w1=0.5, w2=0.5, b=-0.7)


def NAND(x1, x2):
    # (1,1)(x1,x2) - 1.4 = 0 な直線
    return perceptron(x1, x2, w1=-0.5, w2=-0.5, b=0.7)


def OR(x1, x2):
    # (1,1)(x1,x2) - 1.4 = 0 な直線
    return perceptron(x1, x2, w1=0.5, w2=0.5, b=-0.2)


def main(args):
    x = np.arange(-0.1, 1.1, 0.01)
    y = np.arange(-0.1, 1.1, 0.01)

    X, Y = np.meshgrid(x, y)
    Z = XOR_3d(X, Y)

    fig = plt.figure()
    ax = Axes3D(fig)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("f(x, y)")

    # ax.plot_surface(X, Y, Z, alpha=0.3)
    ax.plot_wireframe(X, Y, Z)

    plt.show()


if __name__ == "__main__":
    main(sys.argv)
