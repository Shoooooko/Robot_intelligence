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


def read_data():
    train_img, train_label, test_img, test_label = load_data(
        flatten=False, normalize=True, one_label=True)
    return train_img, train_label, test_img, test_label


def main():
    train_img, train_label, test_img, test_label = read_data()
    net = ConvNet(0.1, input_dim=(1, 28, 28),
                  conv_param={'filter_num': 16,
                              'filter_size': 5, 'pad': 0, 'stride': 1},
                  hidden_num_list=[100, 100], out_num=10)
    epoch = np.arange(10)
    iterations = 10000
    #######
    train_img = train_img[:5000]
    train_label = train_label[:5000]
    train_num = train_img.shape[0]
    batch_num = 200
    dummy_test_img = test_img.flatten()
    dummy_test_img_num = len(dummy_test_img)
    random_ids = [np.random.randint(0, dummy_test_img_num) for i in range(
        int(1 / 4 * dummy_test_img_num))]  # 1/4をd%に合わせてd/100とする
    for i in random_ids:
        dummy_test_img[i] = np.random.random()
    eta = 0.01
    err_list = []
    train_accuracy_list = []
    test_accuracy_list = []
    dummy_accuracy_list = []
    for e in range(len(epoch)):
        for i in range(2000):
            # iter_per_epoch = max(train_num / batch_num, 1)
            batch_id = np.random.choice(train_num, batch_num)
            train_batch = train_img[batch_id]
            answer_batch = train_label[batch_id]
            # 順伝播を計算
            y = predict(train_batch, net)
            # errorの記録
            #error_s = square_error(y, answer_batch)
            error_c = cross_error(y, answer_batch)
            err_list.append(error_c)
            if i % 100 == 0:
                print(i)
            print(error_c)

            # 誤差逆で勾配
            bpropf, net = back_prop(y, answer_batch, net)
            net.params = update_params(bpropf, net.params, eta)

        # 認識精度
        train_accuracy = accuracy_rate(y, answer_batch)
        #print("訓練データに対する正解率", train_accuracy)
        print("train", train_accuracy)
        train_accuracy_list = train_accuracy_list.append(train_accuracy)

        # テストデータの正解率
        # correct_list = []
        data_prediction = predict(test_img, net)
        test_accuracy = accuracy_rate(data_prediction, test_label)
        print("test", test_accuracy)
        test_accuracy_list.append(test_accuracy)

        # dummyでの性能
        dummy_prediction = predict(dummy_test_img, net)
        dummy_test_img = dummy_test_img.reshape(-1, 1, 28, 28)
        dummy_accuracy = accuracy_rate(dummy_prediction, test_label)
        print("dummy", dummy_accuracy)
        dummy_accuracy_list.append(dummy_accuracy)

        if(train_accuracy > 0.95) and (test_accuracy > 0.95):
            exit

    plt.plot(epoch, train_accuracy_list)
    plt.plot(epoch, test_accuracy_list)
    plt.plot(epoch, dummy_accuracy_list)
    # plt.ylim(0, 1.0)
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.show()


if __name__ == '__main__':
    main()
