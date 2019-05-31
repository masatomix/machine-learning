# !/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import numpy as np

W = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
b = np.array([1.0, 2.0, 3.0])


# アフィン変換における数値微分、誤差逆伝播のテスト。

def f(x):
    y = np.dot(x, W) + b
    return y


def loss(x):
    y = f(x)
    return np.sum(y)


## アフィン変換 f: np.dot(x, W) + b
## に対して、L として yを足すという関数を定義。


def main(args):
    x = np.array([[1.0, 2.0],[3.0,5.0]])

    # 数値微分
    grads = numerical_gradients(x)
    print('-----')
    print(grads['x'])
    print(grads['W'])
    print(grads['b'])
    print('-----')

    # 逆伝播
    grads = gradients(x)
    print('-----')
    print(grads['x'])
    print(grads['W'])
    print(grads['b'])
    print('-----')

    # 定義されたクラス
    net = Affine(W, b)
    out = net.forward(x)
    print('--')
    print(out)
    print('--')

    # Lをyで微分した結果を作成して、逆伝播してみる。
    dy = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    back = net.backward(dy)

    print('-----')
    print(back)
    print(net.dW)
    print(net.db)
    print('-----')


def numerical_gradients(x):
    grads = {}

    loss_w = lambda WW: loss(x)
    loss_b = lambda bb: loss(x)
    grads['x'] = numerical_gradient(loss, x)
    grads['W'] = numerical_gradient(loss_w, W)
    grads['b'] = numerical_gradient(loss_b, b)
    return grads


def gradients(x):
    grads = {}
    dy = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    print(dy.shape)
    dx = np.dot(dy, W.T)
    print(dx.shape)
    dw = np.dot(x.T, dy)
    # db = dy  # コレじゃダメ。
    db = np.sum(dy,axis=0)

    grads['x'] = dx
    grads['W'] = dw
    grads['b'] = db
    return grads


def numerical_gradient(f, x):
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x)

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)  # f(x+h)

        x[idx] = tmp_val - h
        fxh2 = f(x)  # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2 * h)

        x[idx] = tmp_val  # 値を元に戻す
        it.iternext()

    return grad


# class MulLayer:
#     def __init__(self):
#         self.x = None
#         self.y = None
#
#     def forward(self, x, y):
#         self.x = x
#         self.y = y
#         return x * y
#
#     def backword(self, dout):
#         dx = dout * self.y
#         dy = dout * self.x
#         return dx, dy
#
#
# class AddLayer:
#     def __init__(self):
#         pass
#
#     def forward(self, x, y):
#         return x + y
#
#     def backword(self, dout):
#         dx = dout * 1
#         dy = dout * 1
#         return dx, dy


class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b

        self.x = None
        self.original_x_shape = None
        # 重み・バイアスパラメータの微分
        self.dW = None
        self.db = None

    def forward(self, x):
        # テンソル対応
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x

        out = np.dot(self.x, self.W) + self.b

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)

        dx = dx.reshape(*self.original_x_shape)  # 入力データの形状に戻す（テンソル対応）
        return dx


if __name__ == "__main__":
    main(sys.argv)
