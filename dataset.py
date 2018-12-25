import urllib.request
import gzip
import pickle
import os
import numpy as np

# data読み込みについて：https://qiita.com/python_walker/items/e4d2ae5b7196cb07402b
# 4種類のdataをfileごとにkeyをつけて読み込む
data_url = 'http://yann.lecun.com/exdb/mnist/'
file_paths = {
    'train_img': 'train-images-idx3-ubyte.gz',
    'train_label': 'train-labels-idx1-ubyte.gz',
    'test_img': 't10k-images-idx3-ubyte.gz',
    'test_label': 't10k-labels-idx1-ubyte.gz'
}
# current_directory
dataset_dir = os.path.dirname(os.path.abspath(__file__))
# pickle形式で保存
save_file = dataset_dir + "/mnist.pkl"

train_num = 60000
test_num = 60000
img_dim = (1, 28, 28)
img_size = 784


def download(file_paths, data_url):
    for path in file_paths.values():
        file_path = dataset_dir + '/' + path
        urllib.request.urlretrieve(data_url + path, file_path)


def load_img(file_name):
    file_path = dataset_dir + '/' + file_name
    with gzip.open(file_path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    data = data.reshape(-1, 784)
    return data


def load_label(file_name):
    file_path = dataset_dir + '/' + file_name
    with gzip.open(file_path, 'rb') as f:
        labels = np.frombuffer(f.read(), np.uint8, offset=8)

    return labels


def setting_data():
    dataset = {}
    dataset['train_img'] = load_img(file_paths['train_img'])
    dataset['train_label'] = load_label(file_paths['train_label'])
    dataset['test_img'] = load_img(file_paths['test_img'])
    dataset['test_label'] = load_label(file_paths['test_label'])
    return dataset


def _normalize(X):
    X = X.astype(np.float32)
    return X/(255.0)


def to_simple_one_label(X):
    #X.shape (60000,)<-train_label (10000,)<-test/label
    T = np.zeros((X.size, 10))
    for col, row in enumerate(T):
        row[X[col]] = 1
    #T.shape (60000,10)<-train_label (10000,10)<-test/label
    return T


def load_data(normalize=True, flatten=True, one_label=True):
    """
    Parameters
    ----------
    normalize : 画像のピクセル値を0.0~1.0に正規化する
    one_hot_label :
        one_hot_labelがTrueの場合、ラベルはone-hot配列として返す
        one-hot配列とは、たとえば[0,0,1,0,0,0,0,0,0,0]のような配列
    flatten : 画像を一次元配列に平にするかどうか
    (訓練画像, 訓練ラベル), (テスト画像, テストラベル)"""
    download(file_paths, data_url)
    dataset = setting_data()
    with open(save_file, 'wb') as f:
        pickle.dump(dataset, f, -1)  # 最も高いprotocolで保存
    # 保存したfileを開く
    with open(save_file, 'rb') as f:
        dataset = pickle.load(f)

    if normalize:
        dataset['train_img'] = _normalize(dataset['train_img'])
        dataset['test_img'] = _normalize(dataset['test_img'])

    if one_label:
        dataset['train_label'] = to_simple_one_label(dataset['train_label'])
        dataset['test_label'] = to_simple_one_label(dataset['test_label'])

    if not flatten:
        for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].reshape(-1, 1, 28, 28)
    with open(save_file, 'wb') as f:
        pickle.dump(dataset, f, -1)  # 最も高いprotocolで保存
    # 保存したfileを開く
    '''with open(save_file, 'rb') as f:
        dataset = pickle.load(f) '''
    return dataset['train_img'], dataset['train_label'], dataset['test_img'], dataset['test_label']


if __name__ == '__main__':
    load_data()
