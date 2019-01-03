import numpy as np


def predict(x, net):
    for layer in net.layers.values():
        x = layer.forward(x)
    return net.final_layer.forward(x)

# 分類の確率がmaxになるところのみしかtestと比較しないので,
    # argmaxでそこのindexのみ用いて精度を確認する


def accuracy_rate(u, t):
    u = np.argmax(u, axis=1)  # axis = 1:列について最大値のindex
    if t.ndim != 1:
        t = np.argmax(t, axis=1)
    accuracy = np.sum(u == t) / float(u.shape[0])
    return accuracy


def back_prop(i, t, net, decay_param=0.1):
    eta = 0.1
    grads = {}
    dy = net.final_layer.backward(t)
    # print('dy')
    # print(dy)
    back_layers = list(net.layers.values())
    back_layers.reverse()
    linear_num = 1
    weight_decay = 0
    for layer in back_layers:
        dy = layer.backward(dy)
        if layer in (net.layers['linear1'], net.layers['linear2'], net.layers['linear3'], net.layers['conv1'], net.layers['conv2']):
              # wx+bの場合は隠れ関数の教会なのでW,Bの勾配を求める
            grads['w' + str(linear_num)] = layer.dW
            grads['b' + str(linear_num)] = layer.dB
            weight_decay = 0.5 * decay_param * np.sum(layer.W ** 2)
            layer.W -= eta*layer.dW + weight_decay
            layer.B -= eta * layer.dB  # バッチ数でわる
            linear_num += 1
    return grads, net


# 重みの更新はMomemtunとかで書き換える


def update_params(gradient, params, eta):
    for param in params.keys():
        params[param] -= eta * gradient[param]
    return params


def cross_error(i, t):
    if i.ndim == 1:  # 行列全部でなくて.data１つあたりの誤差
        t = t.reshape(1, t.size)
        i = i.reshape(1, i.size)
    # 教師データがone-hot-の場合、正解ラベルのインデックスに変換
    if t.size == i.size:
        batch_num = i.shape[0]
    return - np.sum(np.log(i[np.arange(batch_num), t] + 1e-7)) / batch_num


def square_error(i, t):
    return 1 / 2 * np.sum((i - t) ** 2)

    '''weight_decay = 0
        for i in range(1, self.hidden_layer_num + 2):
            W = self.params['W' + str(i)]
            weight_decay += 0.5 * self.weight_decay_lambda * np.sum(W ** 2)

        return self.last_layer.forward(y, t) + weight_decay'''


def map_2d_forward(input_data, filter_h, filter_w, stride=1, pad=0):

    batch_num, depth, in_h, in_w = input_data.shape  # (batch_num, 1, 28, 28)
    out_h = (in_h + 2*pad - filter_h)//stride + 1
    out_w = (in_w + 2 * pad - filter_w) // stride + 1

    # input(高さ、幅部分)にパディング(0埋め)を適用
    after_padding = np.pad(input_data, [(0, 0), (0, 0),
                                        (pad, pad), (pad, pad)], 'constant')
    # out_dataの初期化
    out_data = np.zeros((batch_num, depth, filter_h, filter_w, out_h, out_w))
    # https://www.ibm.com/developerworks/jp/cognitive/librari/cc-convolutional-neural-network-vision-recognition/index.html
    # numpyなしでは上記のリンクのようにfor文4回
    for i in range(filter_h):
        out_i = i + stride*out_h
        for l in range(filter_w):
            out_l = l + stride*out_w
            # out_dataの形状に合わせてinput_data(高さ、幅部分)の値を挿入
            out_data[:, :, i, l, :, :] = after_padding[:,
                                                       :, i: out_i: stride, l: out_l: stride]
    return out_data.transpose(0, 4, 5, 1, 2, 3).reshape(batch_num*out_h*out_w, -1)


# col2imの実装


def map_2d_back(map_out, input_shape, filter_h, filter_w, stride=1, pad=0):
    """
    map_out : 出力データ(2次元配列)
    input_shape : 入力データの形状（例：(10, 1, 28, 28)）
    in_data : 入力データ
    """
    N, C, in_h, in_w = input_shape
    out_h = (in_h + 2*pad - filter_h)//stride + 1
    out_w = (in_w + 2*pad - filter_w)//stride + 1
    # map_outの形状をinput_dataの形状に戻す
    map_out = map_out.reshape(
        N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    # input_dataを初期化
    in_data = np.zeros((N, C, in_h + 2*pad + stride -
                        1, in_w + 2*pad + stride - 1))
    # map_outの値をinput_dataに挿入
    for i in range(filter_h):
        i_max = i + stride*out_h
        for l in range(filter_w):
            l_max = l + stride*out_w
            in_data[:, :, i:i_max:stride,
                    l:l_max:stride] += map_out[:, :, i, l, :, :]

    # パディングを除いて返す
    return in_data[:, :, pad:in_h + pad, pad:in_w + pad]
