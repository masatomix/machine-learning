#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

from matplotlib import pyplot as plt
import numpy as np
from gradient import numerical_gradient
from functions import mean_squared_error


def f(x):
    return (x * 2) + 3.0


def get_data(data_size, train_size=None, train_ratio=0.9, start=-5.0, end=5.0):
    if train_size is None:  # 教師データ数が引数から取れない場合は、
        train_size = int(data_size * train_ratio)  # うち教師データの割合から、教師データ数を計算

    train_index = np.sort(np.random.choice(data_size, train_size, replace=False))  # data から train だけの配列番号をつくる

    # まずdata_sizeコの、全データを作成
    x_all = np.arange(start, end, (end - start) / data_size)  # 数直線を作成。
    t_all = f(x_all)  # 全体のyのデータを算出
    t_all += np.random.normal(0, 0.3, data_size)  # ちょっとだけノイズをたす
    # 全データ作成、以上

    x_train = x_all[train_index]
    t_train = t_all[train_index]
    x_test = np.delete(x_all, train_index)
    t_test = np.delete(t_all, train_index)

    # つくったデータ達は横向きになってるのでreshapeして、縦向きに。
    x_train = x_train.reshape(len(x_train), 1)
    t_train = t_train.reshape(len(t_train), 1)

    # つくったデータ達は横向きになってるのでreshapeして、縦向きに。
    x_test = x_test.reshape(len(x_test), 1)
    t_test = t_test.reshape(len(t_test), 1)

    return (x_train, t_train), (x_test, t_test)


def main(args):
    iteration_count = 50000  # バッチ学習の、繰り返し階数

    data_size = 100  # 母集団のデータ数
    train_size = data_size - 50  # うち教師データ数

    (x_train, t_train), (x_test, t_test) = get_data(data_size, train_size, start=0.0, end=10.0)

    print(f'教師データ数:{x_train.shape}')
    print(f'テストデータ数:{x_test.shape}')

    # print('xの値:' + str(x_train))
    # print('yの値:' + str(t_train))

    # ニューラルネットワークは、1次元で、アウトプットも1次元、隠れ層ナシ
    input_size = 1
    output_size = 1
    network = Net(input_size=input_size, output_size=output_size)

    batch_size = 100  # batch_sizeごとに、無作為抽出してトレーニング開始
    iter_per_epoch = max(train_size / batch_size, 1)
    # t/b で、batch_sizeで処理したときにtrain_sizeを使い切る回数を出している
    # t=10000 b=10 -> t/b = 1000 回=1エポックというらしい。

    print(f'epoch| {iter_per_epoch}')

    # 学習率を設定
    learning_rate = 0.00001

    for i in range(iteration_count):  # batch_sizeコごとのバッチを、iteration_count 回繰り返す
        batch_mask = np.random.choice(train_size, batch_size)  # train_size 数列から、batch_size分、とる
        x_batch = x_train[batch_mask]  # x_trainから、ランダムに選択
        t_batch = t_train[batch_mask]  # t_trainから、ランダムに選択

        # print(x_batch.shape)

        # 勾配ベクトルを算出
        grad = network.numerical_gradient(x_batch, t_batch)

        # パラメータの更新
        for key in ('W1', 'b1'):
            network.network[key] -= learning_rate * grad[key]

        if i % iter_per_epoch == 0:
            loss = network.loss(x_batch, t_batch)
            print(f'train loss ({i})| {str(loss)}')

    print(network.network['W1'])
    print(network.network['b1'])

    print(f'教師データ数:{x_train.shape}')
    print(f'テストデータ数:{x_test.shape}')
    print(f'epoch| {iter_per_epoch}')

    # テスト開始。
    test_loss = network.loss(x_test, t_test)
    print("test loss |" + str(test_loss))

    plt.plot(x_test, t_test, 'o', label='data')
    plt.plot(x_test, network.predict(x_test), label='予測値', linestyle='solid')
    plt.show()


class Net:
    def __init__(self, input_size=None, output_size=None):
        weight_init_std = 0.01
        self.network = {}

        self.network['W1'] = weight_init_std * np.random.randn(input_size, output_size)
        self.network['b1'] = np.zeros(output_size)
        print(self.network['W1'].shape)
        print(self.network['b1'].shape)

    def predict(self, x):
        W1 = self.network['W1']
        b1 = self.network['b1']

        a1 = np.dot(x, W1) + b1
        # z1 = sigmoid(a1)
        # a2 = np.dot(z1, W2) + b2
        # z2 = sigmoid(a2)
        # a3 = np.dot(z2, W3) + b3
        # y = softmax(a3)

        return a1

    def loss(self, x, t):
        y = self.predict(x)
        # print('predict:' + str(y))
        # print('ans:' +str(t))
        return mean_squared_error(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        accuracy_cnt = 0
        for i in range(len(x)):
            # print(f'x: {x[i]}, y:{y[i]} (yは予測値)')
            # print(f't: {t[i]} (yの正解)')
            if y[i] == t[i]:
                accuracy_cnt += 1

        return float(accuracy_cnt) / len(x)

    def numerical_gradient(self, x, t):
        # def loss_W(W):
        #     return self.loss(x, t)

        loss_W = lambda W: self.loss(x, t)
        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.network['W1'])
        # grads['W2'] = numerical_gradient(loss_W, self.network['W2'])
        # grads['W3'] = numerical_gradient(loss_W, self.network['W3'])
        grads['b1'] = numerical_gradient(loss_W, self.network['b1'])
        # grads['b2'] = numerical_gradient(loss_W, self.network['b2'])
        # grads['b3'] = numerical_gradient(loss_W, self.network['b3'])

        # grads['b1'] = np.zeros_like(self.network['b1'])

        return grads


if __name__ == "__main__":
    main(sys.argv)
