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

# (訓練データ, 訓練ラベル, テストデータ, テストラベル)を読み込む
# データは四次元として読み込み, 各ピクセルの画素値は0~1の小数点して正規化


def read_data():
    train_img, train_label, test_img, test_label = load_data(
        flatten=False, normalize=True, one_label=True)
    return train_img, train_label, test_img, test_label


def main():
    train_img, train_label, test_img, test_label = read_data()
    # 畳み込みネットワークのインスタンス生成
    net = ConvNet(0.1, input_dim=(1, 28, 28),
                  conv_param={'filter_num': 16,
                              'filter_size': 5, 'pad': 0, 'stride': 1},
                  hidden_num_list=[100, 100], out_num=10)
    # 10epochを最大とする
    epoch = np.arange(1)
    # 学習回数
    iterations = []
    # データを減らして学習させる場合
    # train_img = train_img[:10000]
    # train_label = train_label[:10000]
    # test_img = test_img[:5000]
    # test_label = test_label[:5000]
    train_num = train_img.shape[0]
    # バッチ数
    batch_num = 5000
    # 学習係数η
    eta = 0.1
    # 誤差
    err_list = []
    # 正解率
    train_accuracy_list = []
    test_accuracy_list = []
    noise_train_accuracy = []
    noise_test_accuracy = []
    noise_train_img = train_img.flatten()
    noise_train_img_num = len(noise_train_img)
    noise_rate = [0, 5, 10, 15, 20, 25]
    # noise_rate dを0~25%まで変化, noise_rate分だけ画素値をランダムに0~1の小数点に変更
    for d in noise_rate:
        # テストデータを一次元にする
        noise_train_img = train_img.flatten()
        random_ids = [np.random.randint(0, noise_train_img_num) for i in range(
            int(d/(100) * noise_train_img_num))]  # d/100のnoiseとする
        for i in random_ids:
            noise_train_img[i] = np.random.random()
            # 入力データをの形状をもとに戻す
        noise_train_img = noise_train_img.reshape(-1, 1, 28, 28)
        for i in range(500):
            # バッチ数分だけ訓練データからランダムにバッチデータのindex選ぶ
            batch_id = np.random.choice(noise_train_img_num, batch_num)
            # バッチindexに対応する学習データとラベル取得
            train_batch = noise_train_img[batch_id]
            answer_batch = train_label[batch_id]
            # 順伝播
            y = predict(train_batch, net)
            # error
            # error_s = square_error(y, answer_batch)#二乗和誤差
            error_c = cross_error(y, answer_batch, net.params)  # 交叉エントロピー誤差
            err_list.append(error_c)

            # 誤差逆伝播法
            bpropf, net = back_prop(y, answer_batch, net)
            # パラメータの更新
            net.params = update_params(bpropf, net.params, eta)
            # 正解率：10iterationsごとに訓練データと学習データに対して求める
            if i % 10 == 0:
                iterations.append(i)
                # 訓練データ
                train_accuracy = accuracy_rate(y, answer_batch)
                train_accuracy_list.append(train_accuracy)
                data_prediction = predict(test_img, net)
                # テストデータ
                test_accuracy = accuracy_rate(data_prediction, test_label)
                test_accuracy_list.append(test_accuracy)
        print("noise", d, "%")
        print("訓練データに対する正解率", train_accuracy)
        noise_train_accuracy.append(train_accuracy)
        print("テストデータに対する正解率", test_accuracy)
        noise_test_accuracy.append(test_accuracy)

        plt.plot(iterations, train_accuracy_list, label='train')
        plt.plot(iterations, test_accuracy_list, label='test')
        #plt.plot(iterations, dummy_accuracy_list)
        # plt.ylim(0, 1.0)
        plt.xlabel('iterations')
        plt.ylabel('accuracy')
        plt.legend(loc='best')
        plt.show()


if __name__ == '__main__':
    main()
