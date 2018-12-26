# u=w*x
# delta=deE/deu
# del_u/del_w=x
# w_new=w_old-eta*grad*xだがbackwordでは出力層からgradの連鎖を求める
import numpy as np


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
        du[self.flag] = 0  # それ以外はduそのまま
        du_x = du
        return du_x


class sigmoid:
    def __init__(self):
        self.u = None

    # 順伝播予測+値のstore
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
        u = np.dot(x, self.W) + self.B
        return u
    # BP
    # u=wx+b=>du_x,du_w,du_bを求める

    def backward(self, du):
        eta = 0.01
        du_x = np.dot(du, self.W.T)
        self.dW = np.dot(self.x.T, du)
        # Bはブロードキャストによって各行全てに影響をもつ、よってまとめる時は和をとる
        self.dB = np.sum(du, axis=0)
        du_x = du_x.reshape(self.x.shape)  # (100.784)
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
            print(du_x, end=" isc Y")
            # データ一個あたりの誤差
        else:
            du_x = self.y.copy()
            du_x[np.arange(batch_num), t] -= 1
            du_x = du_x / batch_num

        return du_x
