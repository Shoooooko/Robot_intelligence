import os
import sys
import pickle
import numpy as np
from collections import OrderedDict
from layers import *
from method import *


class multiLayerNet:
    def __init__(self, input_num, hidden_num_list, out_num, initial_weight):
        self.params = {}
        # 標準正規分布によるmatrix
        self.params['w1'] = np.random.normal(
            0, initial_weight, [input_num, hidden_num_list[0]])
        self.params['w2'] = np.random.normal(
            0, initial_weight, [hidden_num_list[0], hidden_num_list[1]])
        self.params['w3'] = np.random.normal(
            0, initial_weight, [hidden_num_list[1], hidden_num_list[2]])
        self.params['w4'] = np.random.normal(
            0, initial_weight, [hidden_num_list[2], out_num])
        self.params['b1'] = np.zeros(hidden_num_list[0])
        self.params['b2'] = np.zeros(hidden_num_list[1])
        self.params['b3'] = np.zeros(hidden_num_list[2])
        self.params['b4'] = np.zeros(out_num)

        # layersの設定
        # layers_list = {'relu1': relu(), 'relu2': relu(),
        # 'relu3': relu(), 'sigmoid': sigmoid(), 'softmax': softmax()}
        relu1 = relu()
        relu2 = relu()
        relu3 = relu()
        soft1 = softmax()
        self.layers = OrderedDict()
        self.layers['linear1'] = linear(self.params['w1'], self.params['b1'])
        self.layers['relu1'] = relu1
        self.layers['linear2'] = linear(self.params['w2'], self.params['b2'])
        self.layers['relu2'] = relu2
        self.layers['linear3'] = linear(self.params['w3'], self.params['b3'])
        self.layers['relu3'] = relu3
        self.layers['linear4'] = linear(self.params['w4'], self.params['b4'])
        self.final_layer = soft1


class ConvNet:
    """単純なConvNet

    conv - relu - pool - affine - relu - affine - softmax

    Parameters
    ----------
    input_size : 入力サイズ
    hidden_size_list : 隠れ層のニューロンの数のリスト（e.g. [100, 100, 100]）
    output_size : 出力サイズ
    activation : 活性化関数を指定 'relu' or 'sigmoid'
    weight_init_std : 重みの標準偏差を指定（e.g. 0.01）
        'relu'または'he'を指定した場合は「Heの初期値」を設定
        'sigmoid'または'xavier'を指定した場合は「Xavierの初期値」を設定
    """

    # 初期化
    def __init__(self, initial_weight, input_dim=(1, 28, 28),
                 conv_param={'filter_num': 16,
                             'filter_size': 5, 'pad': 0, 'stride': 1},
                 hidden_num_list=[100, 100], out_num=10):
        """
        input_dim : 入力データの次元(チャンネル、高さ、幅)
        conv_param : Convolution層のハイパーパラメーターディクショナリ
            - filter_num : フィルターの数
            - filter_size : フィルターのサイズ
            - pad : パディング
            - stride : ストライド
        hidden_size : 隠れ層(全結合)のニューロンの数
        output_size : 出力層(全結合)のニューロンの数
        initial_weight : 初期化の際の重みの標準偏差
        """

        # Convolution層のハイパーパラメーターをセット
        filter_num = conv_param['filter_num']
        filter_size = conv_param['filter_size']
        filter_pad = conv_param['pad']
        filter_stride = conv_param['stride']
        input_size = input_dim[1]
        # conv_out_size = []
        # pool_out_size = []
        conv_out_size = [(input_size - filter_size + 2 *
                          filter_pad) / filter_stride + 1]  # 28->24 形状は28*28*1->24*24*16
        pool_out_size = [int(
            filter_num * (conv_out_size[0] / 2) * (conv_out_size[0] / 2))]  # 24*24*16->12*12*16
        conv_out_size.append((conv_out_size[0]/2 - filter_size + 2 *
                              filter_pad) / filter_stride + 1)  # 12->8  形状は12*12*16->8*8*16
        pool_out_size.append(int(
            filter_num * (conv_out_size[1] / 2) * (conv_out_size[1] / 2)))  # 4*4*16

        # 重み、バイアスの初期化
        self.params = {}
        # 標準正規分布によるmatrix(out_channel:filter_num, input_channel,hight, width)
        self.params['w1'] = np.random.normal(
            0, np.sqrt(
                2.0/(28*28)),
            [filter_num, input_dim[0], filter_size, filter_size])
        self.params['w2'] = np.random.normal(
            0, np.sqrt(2.0/(24*24)),
            [filter_num, filter_num, filter_size, filter_size])
        self.params['w3'] = np.random.normal(
            0, np.sqrt(2.0/pool_out_size[1]), [pool_out_size[1], hidden_num_list[0]])
        self.params['w4'] = np.random.normal(
            0, np.sqrt(2.0/hidden_num_list[0]), [hidden_num_list[0], hidden_num_list[1]])
        self.params['w5'] = np.random.normal(
            0, np.sqrt(2.0/hidden_num_list[1]), [hidden_num_list[1], out_num])
        self.params['b1'] = np.zeros(filter_num)
        self.params['b2'] = np.zeros(filter_num)
        self.params['b3'] = np.zeros(hidden_num_list[0])
        self.params['b4'] = np.zeros(hidden_num_list[1])
        self.params['b5'] = np.zeros(out_num)

        # レイヤの生成
        self.layers = OrderedDict()  # 順序付きディクショナリ
        self.layers['conv1'] = Convolution(self.params['w1'], self.params['b1'],
                                           conv_param['stride'], conv_param['pad'])
        self.layers['relu1'] = relu()
        self.layers['pool1'] = Pooling(pool_h=2, pool_w=2, stride=2)
        self.layers['conv2'] = Convolution(self.params['w2'], self.params['b2'],
                                           conv_param['stride'], conv_param['pad'])
        self.layers['relu2'] = relu()
        self.layers['pool2'] = Pooling(pool_h=2, pool_w=2, stride=2)

        self.layers['linear1'] = linear(self.params['w3'], self.params['b3'])
        self.layers['relu3'] = relu()
        self.layers['linear2'] = linear(self.params['w4'], self.params['b4'])
        self.layers['relu4'] = relu()
        self.layers['linear3'] = linear(self.params['w5'], self.params['b5'])

        self.final_layer = softmax()
