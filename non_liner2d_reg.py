#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

from matplotlib import pyplot as plt
import numpy as np

from gradient import numerical_gradient
from functions import mean_squared_error, sigmoid, relu
from mpl_toolkits.mplot3d import Axes3D
import pickle


def f(x, y):
    return x * y


def get_data(data_size, train_size=None, train_ratio=0.9, start=-5.0, end=5.0):
    if train_size is None:  # 教師データ数が引数から取れない場合は、
        train_size = int(data_size * train_ratio)  # うち教師データの割合から、教師データ数を計算
    else:
        train_ratio = train_size / data_size

    # まずdata_sizeコの、全データを作成
    x_all = np.arange(start, end, (end - start) / data_size)  # 数直線を作成。
    y_all = np.arange(start, end, (end - start) / data_size)  # 数直線を作成。

    # x_all/y_allでメッシュを作る
    mesh_xs, mesh_ys = np.meshgrid(x_all, y_all)
    # 空の配列を作成
    XX = np.empty(0)
    YY = np.empty(0)
    ZZ = np.empty(0)
    for i, mesh_x in enumerate(mesh_xs):
        # 配列に、ガシガシ足し込んでいく
        XX = np.append(XX, mesh_x)
        YY = np.append(YY, mesh_ys[i])
        ZZ = np.append(ZZ, f(mesh_x, mesh_ys[i]))

    # 実際は、データサイズは二乗だけある。教師データ数は比率で再計算
    data_size = data_size ** 2
    train_size = int(data_size * train_ratio)

    ZZ += np.random.normal(0, 0.3, data_size)  # ちょっとだけノイズをたす
    # 全データ作成、以上

    train_index = np.sort(np.random.choice(data_size, train_size, replace=False))  # data から train だけの配列番号をつくる
    x_train = XX[train_index]
    y_train = YY[train_index]
    t_train = ZZ[train_index]

    x_test = np.delete(XX, train_index)
    y_test = np.delete(YY, train_index)
    t_test = np.delete(ZZ, train_index)

    # つくったデータ達は横向きになってるのでreshapeして、縦向きに。
    x_train = x_train.reshape(-1, 1)  # len(x_train) を指定する意味で  -1
    y_train = y_train.reshape(-1, 1)
    t_train = t_train.reshape(-1, 1)

    # つくったデータ達は横向きになってるのでreshapeして、縦向きに。
    x_test = x_test.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)
    t_test = t_test.reshape(-1, 1)

    X_train = np.append(x_train, y_train, axis=1)
    X_test = np.append(x_test, y_test, axis=1)

    print("教師データ数: ", end='')
    print(X_train.shape)
    print("テストデータ数: ", end='')
    print(X_test.shape)
    return (X_train, t_train), (X_test, t_test)


def init_network():
    with open("network.pkl", 'rb') as f:
        network = pickle.load(f)
    return network


def save_network(x_train, t_train, x_test, t_test):
    train_size = len(x_train)
    iteration_count = 10000  # バッチ学習の、繰り返し階数
    print(f'教師データ数:{x_train.shape}')
    print(f'テストデータ数:{x_test.shape}')

    # print('xの値:' + str(x_train))
    # print('yの値:' + str(t_train))
    # network =  init_network()

    # ニューラルネットワークは、1次元で、アウトプットも1次元、隠れ層ナシ
    input_size = 2
    output_size = 1
    hidden_size = 30
    network = Net(input_size=input_size, hidden_size=hidden_size, output_size=output_size)

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
        # grad = network.gradient(x_batch, t_batch)
        grad = network.numerical_gradient(x_batch, t_batch)

        # print(grad)
        # print(grad2)

        # パラメータの更新
        for key in ('W1', 'b1', 'W2', 'b2'):
            network.network[key] -= learning_rate * grad[key]

        # if i % iter_per_epoch == 0:
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

    with open('network.pkl', 'wb') as f:
        pickle.dump(network, f)


def main(args):
    data_size = 50  # 母集団のデータ数
    train_size = data_size - 1  # うち教師データ数

    # (x_train, t_train), (x_test, t_test) = get_data(data_size, train_ratio=0.9, start=0.0, end=10.0)
    (x_train, t_train), (x_test, t_test) = get_data(data_size, train_size, start=0.0, end=10.0)
    save_network(x_train, t_train, x_test, t_test)

    network = init_network()
    # print(network.predict([1, 6]))
    # plt.plot(x_test, t_test, 'o', label='data')
    # plt.plot(x_test, network.predict(x_test), label='予測値', linestyle='solid')
    # plt.show()

    x = np.arange(0.0, 10.0, 1)
    y = np.arange(0.0, 10.0, 1)

    X, Y = np.meshgrid(x, y)
    # Z = f(X, Y)

    Z2 = np.empty((0, len(X)))
    for i, x in enumerate(X):
        XY = np.stack([x, Y[i]], axis=1)
        Z2_tmp_tmp = network.predict(XY)
        print(Z2_tmp_tmp.shape)
        Z2_tmp = Z2_tmp_tmp.reshape(1, -1)
        print(Z2_tmp.shape)
        Z2 = np.append(Z2, Z2_tmp, axis=0)

    # print(Z.shape)
    # print(Z2.shape)

    fig = plt.figure()
    ax = Axes3D(fig)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("f(x, y)")

    # ax.plot_surface(X, Y, Z, alpha=0.3)
    # ax.plot_surface(X, Y, Z)
    ax.plot_wireframe(X, Y, Z2)

    x_test_T = x_test.T

    print(x_test_T[0].shape)
    print(x_test_T[1].shape)
    print(t_test.T.shape)
    ax.scatter(x_test_T[0], x_test_T[1], t_test.T)

    plt.show()


class Net:
    def __init__(self, input_size=None, hidden_size=None, output_size=None):
        weight_init_std = 0.01
        self.network = {}

        self.network['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.network['b1'] = np.zeros(hidden_size)
        self.network['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.network['b2'] = np.zeros(output_size)
        print(self.network['W1'].shape)
        print(self.network['b1'].shape)

    def predict(self, x):
        W1 = self.network['W1']
        b1 = self.network['b1']
        W2 = self.network['W2']
        b2 = self.network['b2']

        a1 = np.dot(x, W1) + b1
        z1 = relu(a1)
        a2 = np.dot(z1, W2) + b2
        # z2 = relu(a2)
        # a3 = np.dot(z2, W3) + b3
        # y = softmax(a3)

        return a2

    def loss(self, x, t):
        y = self.predict(x)
        # print('---')
        # print('predict:' + str(y))
        # print('ans:' +str(t))
        # print('---')
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
        grads['W2'] = numerical_gradient(loss_W, self.network['W2'])
        # grads['W3'] = numerical_gradient(loss_W, self.network['W3'])
        grads['b1'] = numerical_gradient(loss_W, self.network['b1'])
        grads['b2'] = numerical_gradient(loss_W, self.network['b2'])
        # grads['b3'] = numerical_gradient(loss_W, self.network['b3'])

        # grads['b1'] = np.zeros_like(self.network['b1'])

        return grads

    def gradient(self, x, t):
        y = self.predict(x)
        grads = {}
        grads['W1'] = np.sum((y - t) * x)
        grads['b1'] = np.sum(y - t)

        return grads


if __name__ == "__main__":
    main(sys.argv)
