import sys
# sys.path.append(os.pardir)
import os
from dataset import load_data
from layers import *
from method import *
from net_class import multiLayerNet
import numpy as np
import matplotlib.pylab as plt


# ダミーデータ作成
# train = np.random.rand(100, 784, 10, 10)


def read_data():
    train_img, train_label, test_img, test_label = load_data(
        flatten=True, normalize=True, one_label=True)
    return train_img, train_label, test_img, test_label


def main():
    train_img, train_label, test_img, test_label = read_data()
    train_num = train_img.shape[0]
    # data = train_img[:30]
    # idata = data.reshape(28, 28)
    # img_show(idata)
    batch_num = 6000
    eta = 0.1
    err_list = []
    train_accuracy_list = []
    test_accuracy_list = []
    net = multiLayerNet(input_num=784, hidden_num_list=[
                        100, 100, 100], out_num=10, initial_weight=0.1)
    # noise用
    # noise 10% 入りtest_data
    noise_test_accuracy_list = []
    noise_test_img = test_img.flatten()
    random_id = [np.random.randint(0, 78400)
                 for i in range(int(10 * 78400))]
    # 訓練データへのノイズのrateを変化させてモデルの性能を調べる
    for i in random_id:
        noise_test_img[i] = np.random.random()
    noise_test_img = noise_test_img.reshape(-1, 784)
    ###
    noise_rate = [0, 5, 10, 15, 20, 25]
    iterations = []
    for d in noise_rate:
        train_img = train_img.flatten()
        random_ids = [np.random.randint(0, 78400*6)
                      for i in range(int(d * 78400 * 6))]  # d/100のnoiseとする
        for i in random_ids:
            train_img[i] = np.random.random()
        train_img = train_img.reshape(-1, 784)
    ###
        for i in range(1000):
                # iterations.append(i)
                # バッチ処理設定
            batch_id = np.random.choice(train_num, batch_num)
            train_batch = train_img[batch_id]
            answer_batch = train_label[batch_id]

            # 4層の順伝播を計算
            y = predict(train_batch, net)

            # 誤差逆で勾配
            bpropf, net = back_prop(y, answer_batch, net)
            net.params = update_params(bpropf, net.params, eta)

            # errorの記録
            # error = square_error(y, answer_batch)#二乗和誤差
            error = cross_error(y, answer_batch, net.params)
            err_list.append(error)

        # iterations.append(i)
        # 認識精度
        train_accuracy = accuracy_rate(y, answer_batch)
        train_accuracy_list.append(train_accuracy)

        # テストデータの正解率
        data_prediction = predict(test_img, net)
        test_accuracy = accuracy_rate(data_prediction, test_label)
        test_accuracy_list.append(test_accuracy)

        # ノイズ入りテストデータの正解率
        noise_data_prediction = predict(noise_test_img, net)
        noise_test_accuracy = accuracy_rate(noise_data_prediction, test_label)
        noise_test_accuracy_list.append(noise_test_accuracy)
        print(d, "%")
        print("noise_testデータに対する正解率", noise_test_accuracy)
        print("trainデータに対する正解率", train_accuracy)
        print("testデータに対する正解率", test_accuracy)

    plt.plot(noise_rate, train_accuracy_list, 'o-', label='train')
    plt.plot(noise_rate, test_accuracy_list, 'o-', label='test')
    plt.plot(noise_rate, noise_test_accuracy_list,
             'o-', label='test with noise')
    plt.yticks(np.arange(0.8, 1.11, 0.01))
    plt.ylim(0.8, 1.0)
    plt.xlabel('noise_rate')
    plt.ylabel('accuracy')
    plt.legend(loc='best')
    plt.show()

    # plt.plot(iterations, err_list)
    # #plt.ylim(0, 1.0)
    # plt.show()


if __name__ == '__main__':
    main()
