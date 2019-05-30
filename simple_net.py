#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

import numpy as np
from gradient import numerical_gradient
from functions import cross_entropy_error, softmax


def main(args):
    """
    勾配を求めるサンプル
    :param args:
    :return:
    """
    net = SimpleNet()

    x = np.array([0.6, 0.9])
    p = net.predict(x)
    print(p)
    print(np.argmax(p))

    t = np.array([0, 0, 1])
    print(net.loss(x, t))

    def f(W):
        return net.loss(x, t)

    dW = numerical_gradient(f, net.W)  # この関数は、fにnet.Wを渡して数値微分する関数
    print(dW)


class SimpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3)  # 2x3 の行列

    def predict(self, x):
        a1 = np.dot(x, self.W)
        return a1

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        return cross_entropy_error(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        accuracy_cnt = 0
        for i in range(len(x)):
            # print(f'x: {x[i]}, y:{y[i]} (yは予測値)')
            # print(f't: {t[i]} (yの正解)')
            if y[i] == t[i]:
                accuracy_cnt += 1

        return float(accuracy_cnt) / len(x)

    # def numerical_gradient(self, x, t):
    #     # def loss_W(W):
    #     #     return self.loss(x, t)
    #
    #     loss_W = lambda W: self.loss(x, t)
    #     grads = {}
    #     grads['W1'] = numerical_gradient(loss_W, self.network['W1'])
    #     # grads['W2'] = numerical_gradient(loss_W, self.network['W2'])
    #     # grads['W3'] = numerical_gradient(loss_W, self.network['W3'])
    #     grads['b1'] = numerical_gradient(loss_W, self.network['b1'])
    #     # grads['b2'] = numerical_gradient(loss_W, self.network['b2'])
    #     # grads['b3'] = numerical_gradient(loss_W, self.network['b3'])
    #
    #     # grads['b1'] = np.zeros_like(self.network['b1'])
    #
    #     return grads


if __name__ == "__main__":
    main(sys.argv)
