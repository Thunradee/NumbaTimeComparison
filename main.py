# Author: Thunradee Tangsupakij
# Date: Nov 8th, 2020
# Description: This file compares running time of a machine learning algorithm (PA).
# One uses Numba and the other one don't.
# This file using the MNIST Fashion dataset from https://github.com/zalandoresearch/fashion-mnist.

import numpy as np
import time
import utils.mnist_reader as mnist_reader
from PassiveAggressive import PassiveAggressive
from PassiveAggressive import NumbaPassiveAggressive

def toBinary(y):
    '''
    Convert 0-9 label to even and odd label
    :param y: A one-dimensional array
        0-9 labels
    :return: A one-dimensional array
        even and odd labels; 1 means ood, -1 means even
    '''
    # 1 if odd number, -1 if even number
    bl = [1 if e % 2 else -1 for e in y]
    return np.array(bl)

if __name__ == "__main__":

    # Load MNIST Fashion dataset
    X_train, y_train = mnist_reader.load_mnist('data/fashion', kind='train')
    X_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')

    X_train = X_train.astype(np.float64)
    X_test = X_test.astype(np.float64)
    y_train = toBinary(y_train)
    y_test = toBinary(y_test)

    n_features = len(X_train[0])

    clfs = [(PassiveAggressive(n_features=n_features), "Passive Aggressive"),
            (NumbaPassiveAggressive(n_features=n_features), "Passive Aggressive with Numba")]

    for clf, name in clfs:
        # Training
        start = time.time()
        clf.fit(X_train, y_train)
        end = time.time()
        train_time = end - start

        # Predicting
        start = time.time()
        pred = clf.predict(X_test)
        end = time.time()
        pred_time = end - start

        # Accuracy
        n_mistake = np.sum(pred != y_test)
        n = len(y_test)
        acc = (n - n_mistake) / n

        print(name)
        print("Training time: {} sec".format(train_time))
        print("Predicting time: {} sec".format(pred_time))
        print("Accuracy: {}".format(acc))
        print()
