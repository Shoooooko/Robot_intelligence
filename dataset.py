import urllib.request
import gzip
import pickle
import os
import numpy as np

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

# one_hot_label型にする


def to_simple_one_label(X):
    T = np.zeros((X.size, 10))
    for col, row in enumerate(T):
        row[X[col]] = 1
    return T


def load_data(flatten, normalize=True, one_label=True):
    """
    normalize : 画像のピクセル値を0.0~1.0に正規化する
    one_hot_label :Trueの場合、ラベルはone-hot配列として返す
    flatten : 画像を一次元配列にするかどうか
    """
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
    return dataset['train_img'], dataset['train_label'], dataset['test_img'], dataset['test_label']


if __name__ == '__main__':
    load_data()
