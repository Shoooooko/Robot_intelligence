# https://www.ibm.com/developerworks/jp/cognitive/library/cc-convolutional-neural-network-vision-recognition/index.html
# https://qiita.com/nvtomo1029/items/601af18f82d8ffab551e
# https: // qiita.com/nvtomo1029/items/601af18f82d8ffab551e
# https://qiita.com/ta-ka/items/1c588dd0559d1aad9921


import sys
# sys.path.append(os.pardir)
import os
from dataset import load_data
from layers import *
from method import *
from net_class import ConvNet
import numpy as np
import matplotlib.pylab as plt

# ダミーデータ作成
# train = np.random.rand(100, 784, 10, 10)


def read_data():
    print("data")
    train_img, train_label, test_img, test_label = load_data(
        normalize=True, flatten=True, one_label=True)
    return train_img, train_label, test_img, test_label


def main():
    train_img, train_label, test_img, test_label = read_data()
    print("hoge")
    net = ConvNet(0.1, input_dim=(1, 28, 28),
                  conv_param={'filter_num': 30,
                              'filter_size': 5, 'pad': 0, 'stride': 1},
                  hidden_num_list=[100, 100], output_num=10)
    print("net is created")
    iterations = 10000
    train_num = train_img.shape[0]
    batch_num = 5000
    eta = 0.1
    err_list = []
    accuracy_list = []
    # sgeneralization_list = []
    for i in range(100):
        # iter_per_epoch = max(train_num / batch_num, 1)
        batch_id = np.random.choice(train_num, batch_num)
        train_batch = train_img[batch_id]
        answer_batch = train_label[batch_id]
        # 順伝播を計算
        y = predict(train_batch, net)
        print("predicted!")

##################
    #     # 誤差逆で勾配
    #     bpropf, net = back_prop(y, answer_batch, net)
    #     # net.params = update_params(bpropf, net.params, eta)

    #     # errorの記録
    #     error = square_error(y, answer_batch)
    #     #error = cross_error(y, answer_batch)
    #     err_list.append(error)

    #     # 認識精度
    #     accuracy = accuracy_rate(y, answer_batch)
    #     print("訓練データに対する正解率", accuracy)
    #     # accuracy_list = accuracy_list.append(accuracy)

    # # 認識(推論)を行う

    # def predict(self, x):
    #     for layer in self.layers.values():
    #         x = layer.forward(x)

    #     return x

    # # 損失関数の値を求める
    # # x:入力データ, t:教師データ
    # def loss(self, x, t):
    #     y = self.predict(x)
    #     return self.last_layer.forward(y, t)

    # # 認識精度を求める
    # def accuracy(self, x, t, batch_size=100):
    #     if t.ndim != 1:
    #         t = np.argmax(t, axis=1)

    #     acc = 0.0

    #     for i in range(int(x.shape[0] / batch_size)):
    #         tx = x[i*batch_size:(i+1)*batch_size]
    #         tt = t[i*batch_size:(i+1)*batch_size]
    #         y = self.predict(tx)
    #         y = np.argmax(y, axis=1)
    #         acc += np.sum(y == tt)

    #     return acc / x.shape[0]

    # # 重みパラメータに対する勾配を求める(誤差逆伝搬法)
    # # x:入力データ, t:教師データ
    # def gradient(self, x, t):
    #     """
    #     Returns
    #     -------
    #     各層の勾配を持ったディクショナリ変数
    #         grads['W1']、grads['W2']、...は各層の重み
    #         grads['b1']、grads['b2']、...は各層のバイアス
    #     """
    #     # forward(順伝播)
    #     self.loss(x, t)

    #     # backward(逆伝播)
    #     dout = 1
    #     dout = self.last_layer.backward(dout)

    #     layers = list(self.layers.values())
    #     layers.reverse()
    #     for layer in layers:
    #         dout = layer.backward(dout)

    #     # 求められた勾配値を設定
    #     grads = {}
    #     grads['W1'], grads['b1'] = self.layers['Conv1'].dW, self.layers['Conv1'].db
    #     grads['W2'], grads['b2'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
    #     grads['W3'], grads['b3'] = self.layers['Affine2'].dW, self.layers['Affine2'].db

    #     return grads

    # # パラメータ(重み、バイアス)をファイルに保存する
    # def save_params(self, file_name="params.pkl"):
    #     params = {}
    #     for key, val in self.params.items():
    #         params[key] = val
    #     with open(file_name, 'wb') as f:
    #         pickle.dump(params, f)

    # # ファイルからパラメータ(重み、バイアス)をロードする
    # def load_params(self, file_name="params.pkl"):
    #     with open(file_name, 'rb') as f:
    #         params = pickle.load(f)
    #     for key, val in params.items():
    #         self.params[key] = val

    #     for i, key in enumerate(['Conv1', 'Affine1', 'Affine2']):
    #         self.layers[key].W = self.params['W' + str(i+1)]
    #         self.layers[key].b = self.params['b' + str(i+1)]
