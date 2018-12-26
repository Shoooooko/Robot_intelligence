
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


def back_prop(y, t, net):
    eta = 0.1
    grads = {}
    dy = net.final_layer.backward(t)
    print('dy')
    print(dy)
    back_layers = list(net.layers.values())
    back_layers.reverse()
    linear_num = 1
    for layer in back_layers:
        dy = layer.backward(dy)

        if layer in (net.layers['linear1'], net.layers['linear2'], net.layers['linear3'], net.layers['linear4']):
              # wx+bの場合は隠れ関数の教会なのでW,Bの勾配を求める
            grads['w' + str(linear_num)] = layer.dW
            grads['b' + str(linear_num)] = layer.dB
            # print("before")
            # print(layer.W[0])
            # print(layer.dW[0])
            layer.W -= eta*layer.dW
            layer.B -= eta * layer.dB  # バッチ数でわる
            # print("after")
            # print(layer.W[0])
            # print("hoge")
            linear_num += 1
    #params[param] -= eta * gradient[param]
    return grads, net


# 重みの更新はMomemtunとかで書き換える


def update_params(gradient, params, eta):
    for param in params.keys():
        params[param] -= eta * gradient[param]
    return params


def cross_error(y, t):
    if y.ndim == 1:  # 行列全部でなくて.data１つあたりの誤差
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    # 教師データがone-hot-の場合、正解ラベルのインデックスに変換
    if t.size == y.size:
        batch_num = y.shape[0]
    return - np.sum(np.log(y[np.arange(batch_num), t] + 1e-7)) / batch_num


def square_error(y, t):
    return 1/2*np.sum((y-t)**2)
