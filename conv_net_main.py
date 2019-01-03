# https://www.ibm.com/developerworks/jp/cognitive/library/cc-convolutional-neural-network-vision-recognition/index.html
# https://qiita.com/nvtomo1029/items/601af18f82d8ffab551e
# https: // qiita.com/nvtomo1029/items/601af18f82d8ffab551e
# https://qiita.com/ta-ka/items/1c588dd0559d1aad9921
# https://qiita.com/icoxfog417/items/5fd55fad152231d706c2

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
        flatten=False, normalize=True, one_label=True)
    return train_img, train_label, test_img, test_label


def main():
    train_img, train_label, test_img, test_label = read_data()
    net = ConvNet(0.1, input_dim=(1, 28, 28),
                  conv_param={'filter_num': 16,
                              'filter_size': 5, 'pad': 0, 'stride': 1},
                  hidden_num_list=[100, 100], out_num=10)
    iterations = 10000
    train_num = train_img.shape[0]
    batch_num = 5000
    dammy_test_img = test_img.flatten()
    dammy_test_img_num = len(dammy_test_img)
    random_ids = [np.random.randint(0, dammy_test_img_num) for i in range(
        int(1/4*dammy_test_img_num))]  # 1/4をd%に合わせてd/100とする
    for i in random_ids:
        dammy_test_img[i] = np.random.random()
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

        # errorの記録
        error = square_error(y, answer_batch)
        # error = cross_error(y, answer_batch)
        err_list.append(error)

        # 誤差逆で勾配
        bpropf, net = back_prop(y, answer_batch, net)
        net.params = update_params(bpropf, net.params, eta)

        # 認識精度
        train_accuracy = accuracy_rate(y, answer_batch)
        print("訓練データに対する正解率", train_accuracy)
        # accuracy_list = accuracy_list.append(accuracy)

    # テストデータの正解率
    # correct_list = []
    data_prediction = predict(test_img, net)
    test_accuracy = accuracy_rate(data_prediction, test_label)
    print("testに対する正解率", test_accuracy)

    # dummyでの性能
    dummy_prediction = predict(dammy_test_img, net)
    dummy_accuracy = accuracy_rate(dummy_prediction, test_label)
    print("testに対する正解率", dummy_accuracy)

    x = np.arange(100)
    plt.plot(x, err_list)
    # plt.ylim(0, 1.0)
    plt.xlabel('epoch')
    plt.ylabel('error')
    plt.show()


if __name__ == '__main__':
    main()

    # 認識精度
    accuracy = accuracy_rate(y, answer_batch)
    print("訓練データに対する正解率", accuracy)
    accuracy_list = accuracy_list.append(accuracy)

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
