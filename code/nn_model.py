# ###############################
# Michael Vassernis  -  319582888
#
#################################
import numpy as np
from helper_functions import softmax


class NNModel(object):
    def __init__(self, dims):
        self.dims = dims
        self.params = []
        for i in range(0, len(dims) - 1):
            # using sqrt(6) parameters initialization (Optimal according to an old ML paper)
            r_w = np.sqrt(6) / np.sqrt(dims[i] + dims[i + 1])
            r_b = np.sqrt(6) / np.sqrt(dims[i + 1] + 1)
            w = np.random.uniform(-r_w, r_w, (dims[i], dims[i + 1]))
            b = np.random.uniform(-r_b, r_b, dims[i + 1])
            self.params.append(w)
            self.params.append(b)
        self.num_params = len(self.params)

    def feed_forward(self, input_vec):
        curr_layer = input_vec
        for i in range(0, self.num_params - 2, 2):
            # new layer = old layer * W + b
            curr_layer = np.dot(curr_layer, self.params[i]) + self.params[i + 1]
            # non linearity
            curr_layer = np.tanh(curr_layer)
        curr_layer = np.dot(curr_layer, self.params[-2]) + self.params[-1]
        curr_layer = softmax(curr_layer)
        return curr_layer

    def predict(self, input_vec):
        return np.argmax(self.feed_forward(input_vec))

    def loss_and_gradients(self, input_vec, y_true):
        temp = np.zeros(self.params[-1].shape)
        temp[int(y_true)] = 1
        y_true = temp

        hidden_layers = []
        dropouts = []
        hidden_layers_tanh = [input_vec]
        for i in range(0, self.num_params - 2, 2):
            hidden_layers.append(np.dot(hidden_layers_tanh[i / 2], self.params[i]) + self.params[i + 1])
            dropout = np.random.binomial([np.ones(hidden_layers[i / 2].shape)],1-0.2)[0] * (1.0/(1-0.2))
            dropouts.append(dropout)
            hidden_layers_tanh.append(np.tanh(hidden_layers[i / 2]))  # * dropout)
        y_hat = np.dot(hidden_layers_tanh[len(hidden_layers_tanh) - 1], self.params[-2]) + self.params[-1]
        y_hat = softmax(y_hat)
        loss = - np.sum(y_true * np.log(y_hat))

        d_loss_d_out = -y_true + y_hat * np.sum(y_true)
        grads = []
        for t in range(0, len(hidden_layers_tanh)):
            index = len(hidden_layers_tanh) - 1 - t
            g_b = d_loss_d_out
            g_w = np.outer(hidden_layers_tanh[index], d_loss_d_out)
            grads.insert(0, g_b)
            grads.insert(0, g_w)
            if index > 0:
                d_loss_d_hidden_act = np.dot(self.params[index * 2], d_loss_d_out)
                d_drop_out_loss = dropouts[index - 1]
                d_hidden_act_d_hidden = 1 - np.tanh(hidden_layers[index - 1]) ** 2
                d_loss_d_out = d_loss_d_hidden_act * d_hidden_act_d_hidden * d_drop_out_loss

        return loss, grads

    def train_on_example(self, input_vec, y_true, learning_rate, regularization):
        loss, grads = self.loss_and_gradients(input_vec, y_true)
        for i in range(0, self.num_params):
            # update the parameters with gradients, and add L2 regularization
            self.params[i] -= learning_rate * (grads[i] + self.params[i] * regularization)
        return loss
