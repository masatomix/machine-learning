# !/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import numpy as np

W = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
b = np.array([1.0, 2.0, 3.0])


def main(args):
    # 数値微分
    x = np.array([[1.0, 2.0], [2.0, 3.0]])
    print(loss(affine(x)))
    grads = numerical_gradients(x)

    print('f ここではloss を x_kで微分する')
    print(grads['x'])
    print('f ここではloss を w_ik で微分する')
    print(grads['W'])
    print('f ここではloss を b_iで微分する')
    print(grads['b'])


def affine(vectors):
    y = np.dot(vectors, W) + b
    return y
    # return sigmoid(y)


def loss(vectors):
    # y = affine(_x)
    # print('yShape: ', end='')
    # print(y.shape)
    # return y[0][1]
    # return y[1][0] * 1.0 + y[1][1] * 10.0 + y[1][2] * 100.0
    # print(y)
    # print(np.sum(y))
    return np.sum(vectors)


def numerical_gradients(vectors):
    def loss_x(x):
        y = affine(x)
        return loss(y)

    def loss_w(WW):
        y = affine(vectors)
        return loss(y)

    def loss_b(BB):
        y = affine(vectors)
        return loss(y)

    grads = {}
    grads['x'] = numerical_gradient(loss_x, vectors)
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
