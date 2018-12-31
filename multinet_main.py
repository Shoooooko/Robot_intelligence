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
        normalize=True, flatten=True, one_label=True)
    return train_img, train_label, test_img, test_label


def main():
    train_img, train_label, test_img, test_label = read_data()
    net = multiLayerNet(input_num=784, hidden_num_list=[
                        100, 100, 100], out_num=10, initial_weight=0.1)
    iterations = 10000
    train_num = train_img.shape[0]
    batch_num = 5000
    eta = 0.1
    err_list = []
    accuracy_list = []
    # sgeneralization_list = []
    for i in range(100):
        # print(i)
        #net.params['w1'].shape  (784,100)
        # net.params
        # 1epochあたりの最大の繰りかえし数
        # iter_per_epoch = max(train_num / batch_num, 1)
        batch_id = np.random.choice(train_num, batch_num)
        train_batch = train_img[batch_id]
        answer_batch = train_label[batch_id]
        # 4層の順伝播を計算
        y = predict(train_batch, net)

        # 誤差逆で勾配
        bpropf, net = back_prop(y, answer_batch, net)
        # net.params = update_params(bpropf, net.params, eta)

        # errorの記録
        error = square_error(y, answer_batch)
        #error = cross_error(y, answer_batch)
        err_list.append(error)

        # 認識精度
        accuracy = accuracy_rate(y, answer_batch)
        print("訓練データに対する正解率", accuracy)
        # accuracy_list = accuracy_list.append(accuracy)

    # テストデータの正解率
    #correct_list = []
    data_prediction = predict(test_img, net)
    correct_rate = accuracy_rate(data_prediction, test_label)
    print("testに対する正解率", correct_rate)

    x = np.arange(100)
    plt.plot(x, err_list)
    #plt.ylim(0, 1.0)
    plt.xlabel('epoch')
    plt.ylabel('error')
    plt.show()


if __name__ == '__main__':
    main()
