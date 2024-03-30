from ScribMLP import ScribMLP

from random import random
import os
import struct
import csv
# import pandas as pd
import time
# import cupy as cp
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte'
                                % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte'
                               % kind)

    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',
                                 lbpath.read(8))
        labels = np.fromfile(lbpath,
                             dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack(">IIII",
                                               imgpath.read(16))
        images = np.fromfile(imgpath,
                             dtype=np.uint8).reshape(len(labels), 784)

    return images, labels

import csv
import numpy as np

# X_train, y_train = load_mnist('data/mnist', kind='train')
X_train, y_train = None, None
X_test, y_test = None, None

with open('data/mnist_train.csv', 'r') as read_obj:
    train_set = np.array([list(map(int,rec)) for rec in csv.reader(read_obj, delimiter=',')])
    print(train_set.shape)
    y_train = train_set[:, 0]
    X_train = train_set[:, 1:]

with open('data/mnist_test.csv', 'r') as read_obj:
    test_set = np.array([list(map(int,rec)) for rec in csv.reader(read_obj, delimiter=',')])
    y_test = test_set[:, 0]
    X_test = test_set[:, 1:]

print('Rows: %d, columns: %d' % (X_train.shape[0], X_train.shape[1]))
print('Rows: %d, columns: %d' % (X_test.shape[0], X_test.shape[1]))

nn = ScribMLP(n_output=10,
                  n_features=X_train.shape[1],
                  n_hidden=128,
                  l2=0.1,
                  l1=0.0,
                  epochs=45,
                  eta=0.001,
                  alpha=0.001,
                  decrease_const=0.00001,
                  minibatches=50,
                  random_state=1)


nn.fit(X_train, y_train, print_progress=False)

y_train_pred = nn.predict(X_train)
acc = np.sum(y_train == y_train_pred, axis=0) / X_train.shape[0]
print('Training accuracy: %.2f%%' % (acc * 100))

y_test_pred = nn.predict(X_test)
acc = np.sum(y_test == y_test_pred, axis=0) / X_test.shape[0]
print('Training accuracy: %.2f%%' % (acc * 100))