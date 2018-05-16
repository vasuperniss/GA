# ###############################
# Michael Vassernis  -  319582888
#
#################################
import numpy as np
import struct
import pickle


def softmax(x):
    x -= np.max(x)
    x = np.exp(x)
    x_sum = np.sum(x)
    x /= x_sum
    return x


def read_idx(filename):
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for _ in range(dims))
        return np.fromstring(f.read(), dtype=np.uint8).reshape(shape)


def load_mnist(folder_path):
    print 'loading dataset...'
    train_y = read_idx(folder_path + '/train-labels-idx1-ubyte/data')
    train_x = read_idx(folder_path + '/train-images-idx3-ubyte/data')
    train_x = (train_x.reshape((len(train_y), 28*28)) / 255.0)

    mean = np.sum(train_x) / (len(train_x) * len(train_x[0]))
    std = np.sqrt(np.sum((train_x - mean) ** 2) / (len(train_x) * len(train_x[0])))
    print 'dataset mean:', mean
    print 'dataset std:', std
    train_x = (train_x - mean) # / std
    print 'dataset normalized.'

    train_set = np.c_[train_x.reshape(len(train_x), -1), train_y.reshape(len(train_y), -1)]
    dev_set = train_set[50000:60000]
    train_set = train_set[0:50000]

    np.random.shuffle(train_set)

    return train_set, dev_set


def accuracy_on_dataset(data_set, model):
    correct = wrong = 0.0
    for example in data_set:
        if model.predict(example[:-1]) == example[-1]:
            correct += 1
        else:
            wrong += 1
    return correct / (correct + wrong)


def save_model(model, filename):
    file_pickle = open(filename, 'wb')
    pickle.dump(model, file_pickle)


def load_model(filename):
    file_pickle = open(filename, 'rb')
    model = pickle.load(file_pickle)
    return model
