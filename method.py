import numpy as np

# ネットモデルの各レイヤーに対して順伝播


def predict(x, net):
    for layer in net.layers.values():
        x = layer.forward(x)
    return net.final_layer.forward(x)

# 正解率：予測分類の確率がmaxになるところを正解ラベルと比較する->argmax


def accuracy_rate(u, t):
    u = np.argmax(u, axis=1)  # axis = 1:列について最大値のindex
    if t.ndim != 1:
        t = np.argmax(t, axis=1)
    accuracy = np.sum(u == t) / float(u.shape[0])
    return accuracy

# 誤差逆伝播


def back_prop(i, t, net):
    decay_param = 0.000001  # weight_decayの重み：ヒューリスティックによる値
    grads = {}
    # dy:逆からたどる際の一つ前の層の勾配
    dy = net.final_layer.backward(t)
    back_layers = list(net.layers.values())
    back_layers.reverse()
    #linear_num = 5
    linear_num = 4
    for layer in back_layers:
        dy = layer.backward(dy)
        # 勾配更新
        # if layer in (net.layers['linear1'], net.layers['linear2'], net.layers['linear3'], net.layers['conv1'], net.layers['conv2']):##CNN
        if layer in (net.layers['linear1'], net.layers['linear2'], net.layers['linear3'], net.layers['linear4']):  # Multinet
            grads['w' + str(linear_num)] = layer.dW+decay_param * layer.W
            grads['b' + str(linear_num)] = layer.dB
            linear_num -= 1
    return grads, net


# パラメータの更新
def update_params(gradient, params, eta):
    for param in params.keys():
        params[param] -= eta * gradient[param]
    return params

# 交叉エントロピー法


def cross_error(i, t, params, decay_param=0.000001):
    if i.ndim == 1:  # 行列全部でなく, data１つあたりの誤差
        t = t.reshape(1, t.size)
        i = i.reshape(1, i.size)
    # 教師データがone-hot-の場合、正解ラベルのインデックスに変換
    if t.size == i.size:
        t = t.argmax(axis=1)
    batch_num = i.shape[0]
    # weight_decay
    weight_decay = 0
    # W_sum = np.sum(params['w1']**2)+np.sum(params['w2']**2) + \
    #np.sum(params['w3']** 2) + np.sum(params['w4']** 2) + np.sum(params['w5']** 2)
    W_sum = np.sum(params['w1']**2)+np.sum(params['w2']**2) + \
        np.sum(params['w3']**2)+np.sum(params['w4']**2)
    weight_decay += 0.5 * decay_param * W_sum
    return - np.sum(np.log(i[np.arange(batch_num), t] + 1e-7)) / batch_num + weight_decay

# 二乗和誤差


def square_error(i, t):
    return 1 / 2 * np.sum((i - t) ** 2)

# 順伝播でデータを2次元に展開


def map_2d_forward(input_data, filter_h, filter_w, stride=1, pad=0):
    batch_num, depth, in_h, in_w = input_data.shape  # (batch_num, 1, 28, 28)
    # 出力データの高さと幅
    out_h = (in_h + 2*pad - filter_h)//stride + 1
    out_w = (in_w + 2 * pad - filter_w) // stride + 1

    # 入力データ(高さ、幅)にパディング(値は変更可)を適用
    after_padding = np.pad(input_data, [(0, 0), (0, 0),
                                        (pad, pad), (pad, pad)], 'constant')
    # 出力データの初期化
    out_data = np.zeros((batch_num, depth, filter_h, filter_w, out_h, out_w))
    # https://www.ibm.com/developerworks/jp/cognitive/librari/cc-convolutional-neural-network-vision-recognition/index.html
    # numpyなしでは上記のリンクのようにfor文4回
    for i in range(filter_h):
        out_i = i + stride*out_h
        for l in range(filter_w):
            out_l = l + stride*out_w
            # out_dataの形状に合わせてinput_data(高さ、幅部分)の値を挿入 次元を増やす
            out_data[:, :, i, l, :, :] = after_padding[:,
                                                       :, i: out_i: stride, l: out_l: stride]
    return out_data.transpose(0, 4, 5, 1, 2, 3).reshape(batch_num*out_h*out_w, -1)


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
