# !/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import numpy as np

W = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
b = np.array([1.0, 2.0, 3.0])
# x = np.array([[1.0, 2.0], [1.0, 2.0]])
x = np.array([[1.0, 2.0]])


def main(args):
    # 逆伝播
    grads = gradients(x)
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


def gradients(x):
    dy = np.array([[1.0, 1.0, 1.0]]) # df/dy
    grads = {}
    # print(dy.shape)
    dx = np.dot(dy, W.T)
    # print(dx.shape)
    dw = np.dot(x.T, dy)
    db = dy  # 入力x が多次元の場合はbはコレじゃダメ。
    # db = np.sum(dy, axis=0)

    grads['x'] = dx
    grads['W'] = dw
    grads['b'] = db
    return grads


if __name__ == "__main__":
    main(sys.argv)
