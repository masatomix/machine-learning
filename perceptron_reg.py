#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

from abc import ABCMeta, abstractmethod
import numpy as np


def main(args):

    """
    ニューラルネットワークとしてパーセプトロンを実装したパタン
    :param args:
    :return:
    """
    x_test = np.array([
        [0, 0],
        [1, 0],
        [0, 1],
        [1, 1]
    ])

    # network = AND()
    # print(network.predict(x_test))
    # print('---')
    #
    # network = OR()
    # print(network.predict(x_test))
    # print('---')

    network = XOR()
    print(network.predict(x_test))
    print('---')

    # print(network.network['W1'])
    # print(network.network['b1'])
    # print(network.network['W2'])
    # print(network.network['b2'])


class Gate(metaclass=ABCMeta):

    def h(self, x):
        return step_function(x)

    @abstractmethod
    def predict(self, x):
        pass


class AND(Gate):
    def __init__(self):
        self.network = {}
        self.network['W1'] = np.array([[0.5], [0.5]])
        self.network['b1'] = np.array([-0.7])

    def predict(self, x):
        W1 = self.network['W1']
        b1 = self.network['b1']

        a1 = np.dot(x, W1) + b1
        z1 = super().h(a1)
        return z1


class OR(Gate):
    def __init__(self):
        self.network = {}
        self.network['W1'] = np.array([[0.5], [0.5]])
        self.network['b1'] = np.array([-0.2])

    def predict(self, x):
        W1 = self.network['W1']
        b1 = self.network['b1']

        a1 = np.dot(x, W1) + b1
        z1 = super().h(a1)
        return z1


class NAND(Gate):
    def __init__(self):
        self.network = {}
        self.network['W1'] = np.array([[-0.5], [-0.5]])
        self.network['b1'] = np.array([0.7])

    def predict(self, x):
        W1 = self.network['W1']
        b1 = self.network['b1']

        a1 = np.dot(x, W1) + b1
        z1 = super().h(a1)
        return z1


class XOR(Gate):
    def __init__(self):
        self.network = {}
        self.network['W1'] = np.array([[-0.5, 0.5], [-0.5, 0.5]])
        self.network['b1'] = np.array([0.7, -0.2])

        self.network['W2'] = np.array([[0.5], [0.5]])
        self.network['b2'] = np.array([-0.7])

    def predict(self, x):
        W1 = self.network['W1']
        W2 = self.network['W2']
        b1 = self.network['b1']
        b2 = self.network['b2']

        a1 = np.dot(x, W1) + b1
        z1 = super().h(a1)

        a2 = np.dot(z1, W2) + b2
        z2 = super().h(a2)
        return z2


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


if __name__ == "__main__":
    main(sys.argv)
