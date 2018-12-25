import numpy as np
from collections import OrderedDict
from layers import *
from method import *


class multiLayerNet:
    def __init__(self, input_num, hidden_num_list, out_num, initial_weight):
        self.params = {}
        # 標準正規分布によるmatrix
        self.params['w1'] = initial_weight * \
            np.random.randn(input_num, hidden_num_list[0])
        self.params['b1'] = np.zeros(hidden_num_list[0])
        self.params['w2'] = initial_weight * \
            np.random.randn(hidden_num_list[0], hidden_num_list[1])
        self.params['b2'] = np.zeros(hidden_num_list[1])
        self.params['w3'] = initial_weight * \
            np.random.randn(hidden_num_list[1], hidden_num_list[2])
        self.params['b3'] = np.zeros(hidden_num_list[2])
        self.params['w4'] = initial_weight * \
            np.random.randn(hidden_num_list[2], out_num)
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

    '''def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return self.final_layer.forward(x)
'''
