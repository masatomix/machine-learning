# !/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import numpy as np

W = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
b = np.array([1.0, 2.0, 3.0])


def main(args):
    x = np.array([[1.0, 2.0], [2.0, 3.0]])
    net = Net(W, b)
    print(net.loss(net.predict(x)))
    grads = net.gradient(x)

    print('f ここではloss を x_kで微分する')
    print(grads['x'])
    print('f ここではloss を w_ik で微分する')
    print(grads['W'])
    print('f ここではloss を b_iで微分する')
    print(grads['b'])


class Net:

    def __init__(self, W, b):
        self.layer1 = AffineLayer(W, b)
        self.layer2 = LossLayer()

    def predict(self, x):
        return self.layer1.forward(x)

    def loss(self, y):
        return self.layer2.forward(y)

    def gradient(self, x):
        y = self.predict(x)
        self.loss(y)
        dout = 1
        dout = self.layer2.backward(dout)
        dout = self.layer1.backward(dout)

        grads = {}
        grads['x'] = dout
        grads['W'] = self.layer1.dw
        grads['b'] = self.layer1.db

        return grads


class AffineLayer:
    def __init__(self, W, b):
        self.W = W
        self.b = b

        self.x = None

        # 重み・バイアスパラメータの微分
        self.dw = None
        self.db = None

    def forward(self, x):
        self.x = x
        return np.dot(self.x, self.W) + self.b

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        # print(dx.shape)
        self.dw = np.dot(self.x.T, dout)
        # db = dy  # 入力x が多次元の場合はbはコレじゃダメ。
        self.db = np.sum(dout, axis=0)
        return dx


class LossLayer:
    def __init__(self):
        self.x = None

    def forward(self, x):
        self.x = x
        return np.sum(x)

    def backward(self, dout):
        return np.full(self.x.shape, dout)


if __name__ == "__main__":
    main(sys.argv)
