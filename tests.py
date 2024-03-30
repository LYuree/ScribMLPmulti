from ScribMLP import ScribbleMLP, AFuncs

from random import random
import numpy as np
import csv
# import pandas as pd
import time
# import cupy as cp
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf


mlp = ScribbleMLP([128, 128, 128], 0.05)

# train_set = pd.read_csv('data/mnist_train.csv')
# test_set = pd.read_csv('data/mnist_test.csv')

train_set, test_set = None, None
with open('data/mnist_train.csv', 'r') as read_obj: 
    train_set = [list(map(int,rec)) for rec in csv.reader(read_obj, delimiter=',')]

with open('data/mnist_test.csv', 'r') as read_obj:
    test_set = [list(map(int,rec)) for rec in csv.reader(read_obj, delimiter=',')]

# train_set = train_set[0:20000]
# test_set = test_set[0:3000]

start_time = time.time()
print("\nThe training begins. Please, be patient...\n")
mlp.train(train_set, 3)
print("\n--- %s seconds ---\n" % (time.time() - start_time))
pred_vecs, pred_ints = mlp.predict(test_set)
correct_preds, test_set_size = mlp.get_answer_results()
print(f"\n=============\nTest set size: {test_set_size}, correct answers: {correct_preds}\n===============\n")
print(f"\n=============\nAccuracy: {mlp.get_accuracy()*100:.2f}%\n===============\n")

# inp = np.array(np.random.rand(784, 1) * 256, np.float32)
# inp[0][0] = 5

# array = mlp.predict_one(inp)

# train_item_1 = np.array(np.random.rand(784, 1) * 256, np.float32)
# train_item_2 = np.array(np.random.rand(784, 1) * 256, np.float32)
# train_item_1[0][0] = 5
# train_item_2[0][0] = 6
# train_set = [train_item_1, train_item_2]
# mlp.train(train_set, 2)

# test_item_1 = np.array(np.random.rand(784, 1) * 256, np.float32)
# test_item_2 = np.array(np.random.rand(784, 1) * 256, np.float32)
# test_item_1[0][0] = 5
# test_item_2[0][0] = 6
# test_set = [test_item_1, test_item_2]
# pred_vecs, pred_ints = mlp.predict(test_set)

# print(f"\n{pred_vecs}\n")
# print(f"\n{pred_ints}\n")
# print(f"\n{mlp.get_accuracy():.2f}%\n")