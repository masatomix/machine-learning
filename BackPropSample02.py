# !/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import numpy as np


# def f(x):
#     return x[0] * x[1]


def main(args):
    # print(_numerical_gradient_1d(lambda x: 3 * x, np.array([16.0])))
    # print(numerical_gradient(f, np.array([[2.0], [3.0]])))
    # print(numerical_gradient(f, np.array([2.0, 3.0])))
    print(numerical_gradient(lambda x: (x[0] + x[1]) * x[2], np.array([2.0, 3.0, 4.0])))
    print(gradient(lambda x: (x[0] + x[1]) * x[2], np.array([2.0, 3.0, 4.0])))


def gradient(f, x):
    add = AddLayer()
    addR = add.forward(x[0], x[1])

    mul = MulLayer()
    result = mul.forward(addR, x[2])

    print(result)

    d_mul, d_x2 = mul.backword(1)
    d_x_0, d_x_1 = add.backword(d_mul)

    grad = np.zeros_like(x)
    grad[0] = d_x_0
    grad[1] = d_x_1
    grad[2] = d_x2

    return grad


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


class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y
        return x * y

    def backword(self, dout):
        dx = dout * self.y
        dy = dout * self.x
        return dx, dy


class AddLayer:
    def __init__(self):
        pass

    def forward(self, x, y):
        return x + y

    def backword(self, dout):
        dx = dout * 1
        dy = dout * 1
        return dx, dy


# def _numerical_gradient_1d(f, x):
#     h = 1e-4  # 0.0001
#     grad = np.zeros_like(x)
#
#     for idx in range(x.size):
#         tmp_val = x[idx]
#         x[idx] = float(tmp_val) + h
#         # print(x[idx])
#         fxh1 = f(x[idx])  # f(x+h)
#
#         x[idx] = tmp_val - h
#         # print(x[idx])
#         fxh2 = f(x[idx])  # f(x-h)
#
#         # print(fxh1)
#         # print(fxh2)
#         grad[idx] = (fxh1 - fxh2) / (2 * h)
#
#         x[idx] = tmp_val  # 値を元に戻す
#     return grad


if __name__ == "__main__":
    main(sys.argv)
