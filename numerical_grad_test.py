# !/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import numpy as np
from functions import sigmoid
from gradient import numerical_gradient

W = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
b = np.array([1.0, 2.0, 3.0])


def main(args):
    # 数値微分
    x = np.array([[1.0, 2.0], [2.0, 3.0]])

    print(f'予測値:\n{affine(x)}')
    print(f'損失値: {loss(affine(x))}')

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


if __name__ == "__main__":
    main(sys.argv)
