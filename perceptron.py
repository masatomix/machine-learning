import numpy as np
import sys


def main(args):
    """
    ベタにパーセプトロンを実装したパタン
    :param args:
    :return:
    """
    print('--- and ---')
    print(AND(0, 0))
    print(AND(1, 0))
    print(AND(0, 1))
    print(AND(1, 1))

    print('--- nand ---')
    print(NAND(0, 0))
    print(NAND(1, 0))
    print(NAND(0, 1))
    print(NAND(1, 1))

    print('--- or ---')
    print(OR(0, 0))
    print(OR(1, 0))
    print(OR(0, 1))
    print(OR(1, 1))

    print('--- xor ---')
    print(XOR(0, 0))
    print(XOR(1, 0))
    print(XOR(0, 1))
    print(XOR(1, 1))


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
