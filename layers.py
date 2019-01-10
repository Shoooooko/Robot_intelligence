import numpy as np
from method import *


class relu:
    def __init__(self):
        self.u = None

    def forward(self, x):
        u = x.copy()
        # x>0->false,x<=0->trueとして各要素の情報を保存
        self.flag = (x <= 0)
        u[self.flag] = 0
        return u

    def backward(self, du):
        du[self.flag] = 0
        du_x = du
        return du_x


class sigmoid:
    def __init__(self):
        self.u = None
    # 順伝播予測+値の保持

    def forward(self, x):
        u = 1 / (1 + np.exp(-x))
        # forwardの処理を保存
        self.u = u
        return u
    # 逆伝播の誤差の勾配を求める

    def backward(self, du):
        du_x = du * (1 - self.u) * self.u
        return du_x


class linear:
    def __init__(self, W, B):
        self.W = W
        self.B = B
        self.x = None
        self.dW = None
        self.dB = None

    def forward(self, x):
        self.x = x
        if x.ndim != 2:
            x = x.reshape(-1, x.shape[1]*x.shape[2]*x.shape[3])
        u = np.dot(x, self.W) + self.B
        return u

    def backward(self, du):
        eta = 0.1
        du_x = np.dot(du, self.W.T)
        self.dW = np.dot(self.x.T, du)
        if self.dW.ndim != 2:
            self.dW = self.dW.reshape(-1, self.dW.shape[-1])
        # Bはブロードキャストによって各行全てに影響をもつ、よってまとめる時は和をとる
        self.dB = np.sum(du, axis=0)
        du_x = du_x.reshape(self.x.shape)
        return du_x

# 出力層に用いる-> 入力を正規化して出力->確率として扱える


class softmax:
    def __init__(self):
        self.err = None
        self.y = None

    def softmax(self, x):
        if x.ndim == 2:
            x = x.T
            x = x - np.max(x, axis=0)
            y = np.exp(x) / np.sum(np.exp(x), axis=0)
            return y.T
        x = x - np.max(x)  # オーバーフロー対策
        return np.exp(x) / np.sum(np.exp(x))

    def forward(self, y):
        self.y = self.softmax(y)
        return self.y

    def backward(self, t):
        batch_num = t.shape[0]
        if t.size == self.y.size:  # 教師データがone-hot-vectorのとき
            du_x = (self.y - t) / batch_num
        else:
            du_x = self.y.copy()
            du_x[np.arange(batch_num), t] -= 1
            du_x = du_x / batch_num
        return du_x

# 畳み込み層


class Convolution:
    def __init__(self, W, B, stride=1, pad=0):
        self.W = W
        self.B = B
        self.stride = stride
        self.pad = pad

        # backward時に使用
        self.x = None
        self.map = None  # inputの展開データ
        self.map_W = None  # 重みの展開データ

        # 重み・バイアスパラメータの勾配
        self.dW = None
        self.db = None

    def forward(self, x):
        # filterの個数:filter_num、チャネル数:C、高さ:filter_h、幅:filter_w
        filter_num, C, filter_h, filter_w = self.W.shape
        # input_dataの個数:N、チャネル数:C、高さ:H、幅:W                                                                                                                                N, C, filter_h, filter_w = self.W.shape
        N, C, in_h, in_w = x.shape
        # output_dataの高さ、幅
        out_h = int((in_h + 2*self.pad - filter_h) / self.stride)+1
        out_w = int((in_w + 2*self.pad - filter_w) / self.stride)+1
        # input_dataを2次元配列に展開
        map_x = map_2d_forward(x, filter_h, filter_w, self.stride, self.pad)
        # filterを2次元配列に展開
        map_W = self.W.reshape(filter_num, -1).T
        # out_put
        out = np.dot(map_x, map_W) + self.B
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        self.x = x  # input_data
        self.map = map_x  # input_data(to 2次元)
        self.map_W = map_W  # filter_data(to 2次元)
        return out

    def backward(self, dout):
        filter_num, C, filter_h, filter_w = self.W.shape
        dout = dout.transpose(0, 2, 3, 1).reshape(-1, filter_num)

        self.dB = np.sum(dout, axis=0)  # ブロードキャスト#(16,)
        self.dW = np.dot(self.map.T, dout)
        # 元の4次元になおす
        self.dW = self.dW.transpose(1, 0).reshape(
            self.W.shape)  # conv1: (16,1,5,5),conv2: (16,16,5,5)
        dmap = np.dot(dout, self.map_W.T)
        d_x = map_2d_back(dmap, self.x.shape, filter_h,
                          filter_w, self.stride, self.pad)
        return d_x

# プーリング層


class Pooling:
    def __init__(self, pool_h, pool_w, stride=1, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad

        self.x = None
        self.arg_max = None

    def forward(self, x):
        N, C, H, W = x.shape
        out_h = int((H - self.pool_h) / self.stride + 1)
        out_w = int((W - self.pool_w) / self.stride+1)
        # input_dataをpooling適用領域に2次元展開
        map_x = map_2d_forward(
            x, self.pool_h, self.pool_w, self.stride, self.pad)
        map_x = map_x.reshape(-1, self.pool_h*self.pool_w)
        # map_xの各行に対してmax_pooling
        arg_max = np.argmax(map_x, axis=1)
        out = np.max(map_x, axis=1)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)
        self.x = x
        self.arg_max = arg_max

        return out

    def backward(self, dout):
        dout = dout.transpose(0, 2, 3, 1)

        pool_size = self.pool_h * self.pool_w
        dmax = np.zeros((dout.size, pool_size))
        # 1次元配列にする(flatten)
        dmax[np.arange(self.arg_max.size),
             self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size,))
        dmap_x = dmax.reshape(
            dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = map_2d_back(dmap_x, self.x.shape, self.pool_h,
                         self.pool_w, self.stride, self.pad)

        return dx
