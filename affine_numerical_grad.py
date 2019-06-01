# !/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import numpy as np

W = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
b = np.array([1.0, 2.0, 3.0])
# x = np.array([[1.0, 2.0], [1.0, 2.0]])
x = np.array([[1.0, 2.0]])


def main(args):
    # 数値微分
    grads = numerical_gradients(x)
    print('f ここではloss を x_kで微分する')
    print(grads['x'])
    print('f ここではloss を w_ik で微分する')
    print(grads['W'])
    print('f ここではloss を b_iで微分する')
    print(grads['b'])


def affine(x):
    y = np.dot(x, W) + b
    return y


def loss(x):
    y = affine(x)
    # print('yShape: ', end='')
    # print(y.shape)
    return np.sum(y)
    # return y[0][0] * 1.0 + y[0][1] * 10.0 + y[0][2] * 100.0


def numerical_gradients(x):
    grads = {}

    loss_w = lambda WW: loss(x)
    loss_b = lambda bb: loss(x)
    grads['x'] = numerical_gradient(loss, x)
    grads['W'] = numerical_gradient(loss_w, W)
    grads['b'] = numerical_gradient(loss_b, b)
    return grads


def numerical_gradient(f, x_W_b):
    # print('---')
    # print(x_W_b.shape)
    # print(x_W_b)
    # print('---')
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x_W_b)

    it = np.nditer(x_W_b, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x_W_b[idx]
        x_W_b[idx] = float(tmp_val) + h
        fxh1 = f(x_W_b)  # f(x+h)

        x_W_b[idx] = tmp_val - h
        fxh2 = f(x_W_b)  # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2 * h)

        x_W_b[idx] = tmp_val  # 値を元に戻す
        it.iternext()

    return grad


if __name__ == "__main__":
    main(sys.argv)
