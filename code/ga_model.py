# ###############################
# Michael Vassernis  -  319582888
#
#################################
import numpy as np
import random as rand
from helper_functions import softmax


class Genetic_NNModel(object):
    def __init__(self, dims):
        self.dims = dims
        self.batch_loss = 0.0
        self.params = []
        self.init_distribution = []
        for i in range(0, len(dims) - 1):
            # using sqrt(6) parameters initialization (Optimal according to an old ML paper)
            rW = np.sqrt(6) / np.sqrt(dims[i] + dims[i + 1])
            rb = np.sqrt(6) / np.sqrt(dims[i + 1] + 1)
            self.init_distribution.append(rW)
            self.init_distribution.append(rb)
            W = np.random.uniform(-rW, rW, (dims[i + 1], dims[i]))
            b = np.random.uniform(-rb, rb, dims[i + 1])
            self.params.append(W)
            self.params.append(b)
        self.num_params = len(self.params)

    def feed_forward(self, input_vec):
        curr_layer = input_vec
        keep = 1.0 - 0.3
        scale = 1.0 / keep
        for i in range(0, self.num_params - 2, 2):
            # new layer = old layer * W + b
            curr_layer = np.dot(self.params[i], curr_layer) + self.params[i + 1]
            # non linearity
            dropout = np.random.binomial(1, keep, size=curr_layer.shape) * scale
            curr_layer = np.tanh(curr_layer) * dropout
            # curr_layer = np.maximum(curr_layer, 0, curr_layer)
        curr_layer = np.dot(self.params[-2], curr_layer) + self.params[-1]
        curr_layer = softmax(curr_layer)
        return curr_layer

    def predict(self, input_vec):
        curr_layer = input_vec
        for i in range(0, self.num_params - 2, 2):
            # new layer = old layer * W + b
            curr_layer = np.dot(self.params[i], curr_layer) + self.params[i + 1]
            # non linearity
            curr_layer = np.tanh(curr_layer)
            # curr_layer = np.maximum(curr_layer, 0, curr_layer)
        curr_layer = np.dot(self.params[-2], curr_layer) + self.params[-1]
        # curr_layer = softmax(curr_layer)
        return np.argmax(curr_layer)

    def loss(self, input_vec, y_true):
        y_hat = self.feed_forward(input_vec)
        loss = - np.log(y_hat[int(y_true)])
        return loss

    def loss_on_batch(self, batch):
        self.batch_loss = 0.0
        for example in batch:
            self.batch_loss += self.loss(example[:-1], example[-1])
        return self.batch_loss

    def mate_with(self, model2, blank):
        child = blank
        for i in range(self.num_params):
            param1 = self.params[i]
            param2 = model2.params[i]
            if len(param2.shape) > 1:
                dim, size = param2.shape
            else:
                dim = param2.shape[0]
                size = 1
            for j in range(dim):
                flip = rand.randint(0, 1)
                if flip == 0:
                    child.params[i][j] = param1[j]
                else:
                    child.params[i][j] = param2[j]
            if rand.randint(0, 2) >= 1:
                child.params[i] += np.random.normal(0, 0.001, child.params[i].shape)
            if rand.randint(0, 9) == 1:
                x = rand.randint(0, dim-1)
                y = rand.randint(0, size-1)
                r = self.init_distribution[i]
                if size > 1:
                    child.params[i][x, y] = np.random.uniform(-r, r)
                elif rand.randint(0, 1) == 1:
                    child.params[i][x] = np.random.uniform(-r, r)
        return child