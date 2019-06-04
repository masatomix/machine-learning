# !/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import numpy as np

W = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
b = np.array([1.0, 2.0, 3.0])


def main(args):
    # 逆伝播
    x = np.array([[1.0, 2.0], [2.0, 3.0]])
    print(loss(affine(x)))
    grads = gradients(x)

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


def gradients(vectors):
    y = affine(vectors)
    dy = np.full(y.shape, 1.0)  # df/dy
    # dy = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])  # df/dy
    grads = {}
    # print(dy.shape)
    dx = np.dot(dy, W.T)
    # print(dx.shape)
    dw = np.dot(vectors.T, dy)
    # db = dy  # 入力x が多次元の場合はbはコレじゃダメ。
    db = np.sum(dy, axis=0)

    grads['x'] = dx
    grads['W'] = dw
    grads['b'] = db
    return grads


if __name__ == "__main__":
    main(sys.argv)
