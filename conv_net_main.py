# https://www.ibm.com/developerworks/jp/cognitive/library/cc-convolutional-neural-network-vision-recognition/index.html
# https://qiita.com/nvtomo1029/items/601af18f82d8ffab551e
# https: // qiita.com/nvtomo1029/items/601af18f82d8ffab551e
# https://qiita.com/ta-ka/items/1c588dd0559d1aad9921
# https://qiita.com/icoxfog417/items/5fd55fad152231d706c2
# https://qiita.com/yakof11/items/7c27ae617651e76f03ca
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
    epoch = np.arange(10)
    # 学習回数
    iterations = []
    # データを減らして学習させる場合
    # train_img = train_img[:5000]
    # train_label = train_label[:5000]
    # test_img = test_img[:1000]
    # test_label = test_label[:1000]
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
    dummy_accuracy_list = []

    for e in range(len(epoch)):
        for i in range(1000):
            # バッチ数分だけ訓練データからランダムにバッチデータのindex選ぶ
            batch_id = np.random.choice(train_num, batch_num)
            # バッチindexに対応する学習データとラベル取得
            train_batch = train_img[batch_id]
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

        print("訓練データに対する正解率", train_accuracy)
        #print("train", train_accuracy)

        print("テストデータに対する正解率", test_accuracy)
        # test_accuracy_list.append(test_accuracy)

        plt.plot(iterations, train_accuracy_list, 'o-', label='train')
        plt.plot(iterations, test_accuracy_list, 'o-', label='test')
        #plt.plot(iterations, dummy_accuracy_list)
        # plt.ylim(0, 1.0)
        plt.xlabel('iterations')
        plt.ylabel('accuracy')
        plt.legend(loc='best')
        plt.show()
        # 正解率が訓練/テストデータともに0.95を超えているかチェック
        if(train_accuracy > 0.95) and (test_accuracy > 0.95):
            print("yes")
        # テストデータを一次元にする
        dummy_test_img_num = len(dummy_test_img)
        dummy_rate = []
        # dummyの率dを0~25%まで変化, dummy分だけ画素値をランダムに0~1の小数点に変更
        for d in range(26):
            dummy_test_img = test_img.flatten()
            dummy_rate.append(d)
            random_ids = [np.random.randint(0, dummy_test_img_num) for i in range(
                int(d/(100) * dummy_test_img_num))]  # d/100のダミーとする
            for i in random_ids:
                dummy_test_img[i] = np.random.random()
            # 入力データをの形状をもとに戻す
            dummy_test_img = dummy_test_img.reshape(-1, 1, 28, 28)
            dummy_prediction = predict(dummy_test_img, net)
            # ダミーデータについても予測正解率を求める
            dummy_accuracy = accuracy_rate(dummy_prediction, test_label)
            print("ダミーデータに対する正解率", dummy_accuracy)
            dummy_accuracy_list.append(dummy_accuracy)
        plt.plot(dummy_rate, dummy_accuracy_list)
        plt.show()


if __name__ == '__main__':
    main()
