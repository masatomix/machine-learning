#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys


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


def main(args):
    mul_layer = AddLayer()
    apple = 100
    apple_num = 2
    apple_price = mul_layer.forward(apple, apple_num)

    print(apple_price)
    print(mul_layer.backword(1))


if __name__ == "__main__":
    main(sys.argv)
