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
        curr_layer = input_vec.T
        for i in range(0, self.num_params - 2, 2):
            # new layer = old layer * W + b
            curr_layer = (np.dot(self.params[i], curr_layer).T + self.params[i + 1]).T
            # non linearity
            curr_layer = np.tanh(curr_layer)
            # curr_layer = np.maximum(curr_layer, 0, curr_layer)
            # mean = np.sum(curr_layer) / (len(curr_layer) * len(curr_layer[0]))
            # std = np.sqrt(np.sum((curr_layer - mean) ** 2) / (len(curr_layer) * len(curr_layer[0])))
            # curr_layer = (curr_layer - mean) / std
        curr_layer = np.dot(self.params[-2], curr_layer).T + self.params[-1]
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
            # mean = np.sum(curr_layer) / (len(curr_layer))
            # std = np.sqrt(np.sum((curr_layer - mean) ** 2) / (len(curr_layer)))
            # curr_layer = (curr_layer - mean) / std
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

    def loss_on_batch2(self, batch):
        self.batch_loss = 0.0
        y_hats = self.feed_forward(batch[:,:-1])
        self.batch_loss = -np.sum((np.log(y_hats[np.arange(len(y_hats)), batch[:,-1].astype(dtype=int)])))
        return self.batch_loss


    def mate_with(self, model2, blank, avg_rank):
        child = blank
        for i in range(0, self.num_params, 1):
            param1 = self.params[i]
            param2 = model2.params[i]
            if len(param2.shape) > 1:
                dim, size = param2.shape
            else:
                dim = param2.shape[0]
                size = 1
            # if size == 1:
            if size > 1 and rand.randint(0, 40) == -1:
                for j in range(size):
                    flip = rand.randint(0, 1)
                    if flip == 0:
                        child.params[i][:,j] = param1[:,j]
                    else:
                        child.params[i][:,j] = param2[:,j]
            else:
                # for j in range(dim):
                #     if size > 1 and rand.randint(0, 500) == 0:
                #         r = self.init_distribution[i]
                #         child.params[i][j] =  np.random.uniform(-r, r, (size))
                #     else:
                #         flip = rand.randint(0, 1)
                #         if flip == 0:
                #             child.params[i][j] = param1[j]
                #         else:
                #             child.params[i][j] = param2[j]
                D = np.random.binomial(1, 0.5, size=dim)
                child.params[i] = param1 * D[:, np.newaxis] + param2 * (1 - D)
            if rand.randint(0, 4) >= 1:
                if avg_rank <= 4:
                    child.params[i] += np.random.normal(0, 0.00001, child.params[i].shape)
                if avg_rank <= 10:
                    child.params[i] += np.random.normal(0, 0.00005, child.params[i].shape)
                else:
                    child.params[i] += np.random.normal(0, 0.0001, child.params[i].shape)
            for rank in range(avg_rank / 2):
                if rand.randint(0, 4) == 1:
                    x = rand.randint(0, dim-1)
                    y = rand.randint(0, size-1)
                    r = self.init_distribution[i] * 3
                    if size > 1:
                        child.params[i][x, y] = np.random.uniform(-r, r)
                    elif rand.randint(0, 1) == 1:
                        child.params[i][x] = np.random.uniform(-r, r)
        return child
