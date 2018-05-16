# ###############################
# Michael Vassernis  -  319582888
#
#################################
import numpy as np
import random as rand
from helper_functions import softmax


class Genetic_NNModel(object):
    def __init__(self, dims, init_with_zeros=False):
        self.dims = dims
        self.batch_loss = 0.0
        self.params = []
        self.init_distribution = []
        if not init_with_zeros:
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
        else:
            # self.dims = dims.dims
            # for param in dims.params:
            #    self.params.append(param.copy())
            for i in range(0, len(dims) - 1):
                W = np.zeros((dims[i + 1], dims[i]))
                b = np.zeros(dims[i + 1])
                self.params.append(W)
                self.params.append(b)
        self.num_params = len(self.params)

    def feed_forward(self, input_vec):
        curr_layer = input_vec
        for i in range(0, self.num_params - 2, 2):
            # new layer = old layer * W + b
            curr_layer = np.dot(self.params[i], curr_layer) + self.params[i + 1]
            # non linearity
            # curr_layer = np.tanh(curr_layer)
            curr_layer = np.maximum(curr_layer, 0, curr_layer)
        curr_layer = np.dot(self.params[-2], curr_layer) + self.params[-1]
        curr_layer = softmax(curr_layer)
        return curr_layer

    def predict(self, input_vec):
        return np.argmax(self.feed_forward(input_vec))

    def loss(self, input_vec, y_true):
        y_hat = self.feed_forward(input_vec)
        loss = - np.log(y_hat[y_true])
        return loss

    def loss_on_batch(self, batch_X, batch_Y):
        self.batch_loss = 0.0
        batch_size = len(batch_X)
        for i in range(batch_size):
            self.batch_loss += self.loss(batch_X[i], batch_Y[i])

    def mate_with(self, model2, blank):
        # child = NNModel_Reversed(self.dims, True)
        # child = NNModel_Reversed(self, True)
        child = blank
        for i in range(self.num_params):
            # print child.params[i]
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
            # print dim, size
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
            # print child.params[i]
        return child

    def loss_and_gradients(self, input_vec, y_true):
        temp = np.zeros(self.params[-1].shape)
        temp[y_true] = 1
        y_true = temp

        hiddens = []
        dropouts = []
        hiddens_tanh = [input_vec]
        for i in range(0, self.num_params - 2, 2):
            hiddens.append(np.dot(hiddens_tanh[i / 2], self.params[i]) + self.params[i + 1])
            # dropout = np.random.binomial([np.ones(hiddens[i / 2].shape)],1-0.5)[0] * (1.0/(1-0.5))
            # dropouts.append(dropout)
            hiddens_tanh.append(np.tanh(hiddens[i / 2]))  # * dropout)
        y_hat = np.dot(hiddens_tanh[len(hiddens_tanh) - 1], self.params[-2]) + self.params[-1]
        y_hat = softmax(y_hat)
        loss = - np.sum(y_true * np.log(y_hat))

        d_loss_d_out = -y_true + y_hat * np.sum(y_true)
        grads = []
        for t in range(0, len(hiddens_tanh)):
            index = len(hiddens_tanh) - 1 - t
            gb = d_loss_d_out * 1
            # rows, cols = params[index * 2].shape
            # gW = np.zeros((rows, cols))
            # for i in range(0, rows):
            #     gW[i] = hiddens_tanh[index][i] * d_loss_d_out
            gW = np.outer(hiddens_tanh[index], d_loss_d_out)
            grads.insert(0, gb)
            grads.insert(0, gW)
            if index > 0:
                d_loss_d_hidden_act = np.dot(self.params[index * 2], d_loss_d_out)
                # d_drop_out_loss = dropouts[index - 1]
                d_hidden_act_d_didden = 1 - np.tanh(hiddens[index - 1]) ** 2
                d_loss_d_out = d_loss_d_hidden_act * d_hidden_act_d_didden  # * d_drop_out_loss

        return loss, grads

    def train_on_example(self, input_vec, y_true, learning_rate):
        loss, grads = self.loss_and_gradients(input_vec, y_true)
        for i in range(0, self.num_params):
            # update the parameters with gradients, and add L2 regularization
            self.params[i] -= grads[i] * learning_rate + self.params[i] * (learning_rate * 0.0001)
        return loss

    def loss_and_gradients_batch(self, input_vec_batch, y_true_batch, batch_size, learning_rate):
        # losses = 0.0
        # gradses = []
        loss, grads = self.loss_and_gradients(input_vec_batch[0], y_true_batch[0])
        for i in range(1, batch_size, 1):
            l, g = self.loss_and_gradients(input_vec_batch[i], y_true_batch[i])
            loss += l
            grads += g
        return loss, grads