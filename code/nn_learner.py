import nn_model as mdl
import numpy as np
import struct


def read_idx(filename):
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.fromstring(f.read(), dtype=np.uint8).reshape(shape)


def accuracy_on_dataset(data_X, data_Y, model):
    good = bad = 0.0
    l = len(data_Y)
    for i in range(l):
        if model.predict(data_X[i]) == data_Y[i]:
            good += 1
        else:
            bad += 1
    return good / (good + bad)


def train_classifier(train_X, train_Y, dev_X, dev_Y, num_iterations, learning_rate, model):
    for I in xrange(num_iterations):
        print '\nbegan doing epoch no.', I + 1
        total_loss = 0.0  # total loss in this iteration.
        avg_loss = 0.0
        l = len(train_Y)
        for i in range(l):
            loss = model.train_on_example(train_X[i], train_Y[i], learning_rate)
            total_loss += loss
            avg_loss += loss
            if (i + 1) % 5000 == 0:
                print '\t', (i + 1) / float(l), '% complete. average loss <-', avg_loss / 5000
                avg_loss = 0
        train_loss = total_loss / len(train_X)
        train_accuracy = accuracy_on_dataset(train_X, train_Y, model)
        dev_accuracy = accuracy_on_dataset(dev_X, dev_Y, model)
        print I, train_loss, train_accuracy, dev_accuracy


if __name__ == '__main__':
    model = mdl.NNModel([28*28, 200, 10])

    train_Y = read_idx('mnist_data/train-labels-idx1-ubyte/data')
    train_X = read_idx('mnist_data/train-images-idx3-ubyte/data')
    train_X = (train_X.reshape((len(train_Y), 28*28)) / 255.0 - 0.1307) / 0.3081

    dev_X = train_X[50000:60000]
    dev_Y = train_Y[50000:60000]

    train_X = train_X[0:50000]
    train_Y = train_Y[0:50000]

    # pari_X = np.zeros((5000, 30))
    # pari_Y = np.zeros(5000, dtype=int)
    #
    # pari_d_X = np.zeros((5000, 30))
    # pari_d_Y = np.zeros(5000, dtype=int)
    #
    # for i in range(5000):
    #     a = np.random.choice([0, 1], size=(30,), p=[1./2, 1./2])
    #     res = int(np.sum(a))
    #     if res % 2 == 0:
    #         res = 1
    #     else:
    #         res = 0
    #     pari_X[i] = a
    #     pari_Y[i] = res
    #
    # for i in range(5000):
    #     a = np.random.choice([0, 1], size=(30,), p=[1./2, 1./2])
    #     res = int(np.sum(a))
    #     if res % 2 == 0:
    #         res = 1
    #     else:
    #         res = 0
    #     pari_d_X[i] = a
    #     pari_d_Y[i] = res
    #
    # train_classifier(pari_X, pari_Y, pari_d_X, pari_d_Y, 1000, 0.01, model)
    train_classifier(train_X, train_Y, dev_X, dev_Y, 20, 0.01, model)
