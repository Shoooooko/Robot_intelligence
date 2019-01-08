import sys
# sys.path.append(os.pardir)
import os
from dataset import load_data
from layers import *
#from PIL import Image
from method import *
from net_class import multiLayerNet
import numpy as np
import matplotlib.pylab as plt


# ダミーデータ作成
# train = np.random.rand(100, 784, 10, 10)


# def img_show(img):
#     pil_img = Image.fromarray(np.uint8(img))
#     pil_img.show()


def read_data():
    train_img, train_label, test_img, test_label = load_data(
        flatten=True, normalize=True, one_label=True)
    return train_img, train_label, test_img, test_label


def main():
    train_img, train_label, test_img, test_label = read_data()
    train_num = train_img.shape[0]
    # img_show(idata)
    random_ids = [np.random.randint(0, train_num) for i in range(
        int(20/(100) * train_num))]  # d/100のnoiseとする
    for i in random_ids:
        train_img[i] = np.random.random()
    batch_num = 5000
    eta = 0.1
    err_list = []
    train_accuracy_list = []
    test_accuracy_list = []

    noise_rate = [50, 100, 150, 200]
    for d in noise_rate:
        # # テストデータを一次元にする
        # random_ids = [np.random.randint(0, train_num) for i in range(
        #     int(d/(100) * train_num))]  # d/100のnoiseとする
        # for i in random_ids:
        #     train_img[i] = np.random.random()

        net = multiLayerNet(input_num=784, hidden_num_list=[
                            d, d, d], out_num=10, initial_weight=0.1)
        epochs = 10
        iterations = []

        for i in range(1000):
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
            net.params = update_params(bpropf, net.params, eta)

            # errorの記録
            #error = square_error(y, answer_batch)
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

        print("trainデータに対する正解率", train_accuracy)
        print("testデータに対する正解率", test_accuracy)

    plt.plot(noise_rate, train_accuracy_list, 'o-', label='train')
    plt.plot(noise_rate, test_accuracy_list, 'o-', label='test')
    plt.ylim(0, 1.0)
    plt.xlabel('neuron_number')
    plt.ylabel('accuracy')
    plt.legend(loc='best')
    plt.show()

    # plt.plot(iterations, err_list)
    # #plt.ylim(0, 1.0)
    # plt.show()


if __name__ == '__main__':
    main()
