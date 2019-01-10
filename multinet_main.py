import sys
# sys.path.append(os.pardir)
import os
from dataset import load_data
from layers import *
from method import *
from net_class import multiLayerNet
import numpy as np
import matplotlib.pylab as plt


def read_data():
    train_img, train_label, test_img, test_label = load_data(
        flatten=True, normalize=True, one_label=True)
    return train_img, train_label, test_img, test_label


def main():
    # データ読み込み
    train_img, train_label, test_img, test_label = read_data()
    train_num = train_img.shape[0]
    batch_num = 5000
    eta = 0.1
    err_list = []
    train_accuracy_list = []
    test_accuracy_list = []
    net = multiLayerNet(input_num=784, hidden_num_list=[
                        100, 100, 100], out_num=10, initial_weight=0.1)
    iterations = []

    for i in range(10):
        iterations.append(i)
        # バッチ処理設定
        batch_id = np.random.choice(train_num, batch_num)
        train_batch = train_img[batch_id]
        answer_batch = train_label[batch_id]
        # 順伝播
        y = predict(train_batch, net)

        # 誤差逆伝播
        bpropf, net = back_prop(y, answer_batch, net)
        # パラメータ更新
        net.params = update_params(bpropf, net.params, eta)

        # errorの記録
        # error = square_error(y, answer_batch)#二乗和誤差
        error = cross_error(y, answer_batch, net.params)  # 交叉エントロピー誤差
        err_list.append(error)

        # 訓練データの正解率
        train_accuracy = accuracy_rate(y, answer_batch)
        train_accuracy_list.append(train_accuracy)

        # テストデータの正解率
        data_prediction = predict(test_img, net)
        test_accuracy = accuracy_rate(data_prediction, test_label)
        test_accuracy_list.append(test_accuracy)

    print("trainデータに対する正解率", train_accuracy)
    print("testデータに対する正解率", test_accuracy)

    plt.plot(iterations, train_accuracy_list, 'o-', label='train')
    plt.plot(iterations, test_accuracy_list, 'o-', label='test')
    plt.ylim(0, 1.0)
    plt.xlabel('iterations')
    plt.ylabel('accuracy')
    plt.legend(loc='best')
    plt.show()

    # plt.plot(iterations, err_list)
    # #plt.ylim(0, 1.0)
    # plt.show()


if __name__ == '__main__':
    main()
