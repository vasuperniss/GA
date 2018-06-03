# ###############################
# Michael Vassernis  -  319582888
#
#################################
import numpy as np
from helper_functions import softmax, softmax_batch


class NNModel(object):
    def __init__(self):
        self.batch_loss = 0.0
        self.w1 = None
        self.b1 = None

        self.w2 = None
        self.b2 = None

        self.w3 = None
        self.b3 = None

    def initialize(self, input_dim, hidden1_dim, hidden2_dim, output_dim):
        # using sqrt(6) parameters initialization (Optimal according to an old ML paper)
        r_w = np.sqrt(6) / np.sqrt(input_dim + hidden1_dim)
        r_b = np.sqrt(6) / np.sqrt(hidden1_dim)
        self.w1 = np.random.uniform(-r_w, r_w, (input_dim, hidden1_dim))
        self.b1 = np.random.uniform(-r_b, r_b, hidden1_dim)

        r_w = np.sqrt(6) / np.sqrt(hidden1_dim + hidden2_dim)
        r_b = np.sqrt(6) / np.sqrt(hidden2_dim)
        self.w2 = np.random.uniform(-r_w, r_w, (hidden1_dim, hidden2_dim))
        self.b2 = np.random.uniform(-r_b, r_b, hidden2_dim)

        r_w = np.sqrt(6) / np.sqrt(hidden2_dim + output_dim)
        r_b = np.sqrt(6) / np.sqrt(output_dim)
        self.w3 = np.random.uniform(-r_w, r_w, (hidden2_dim, output_dim))
        self.b3 = np.random.uniform(-r_b, r_b, output_dim)

    def feed_forward(self, input_vec):
        curr_layer = np.dot(input_vec, self.w1) + self.b1
        curr_layer = np.maximum(curr_layer, 0, curr_layer)  # np.tanh(curr_layer)

        curr_layer = np.dot(curr_layer, self.w2) + self.b2
        curr_layer = np.maximum(curr_layer, 0, curr_layer)  # np.tanh(curr_layer)

        curr_layer = np.dot(curr_layer, self.w3) + self.b3
        return curr_layer

    def predict(self, input_vec):
        return np.argmax(self.feed_forward(input_vec))

    def loss(self, input_vec, y_true):
        y_hat = softmax(self.feed_forward(input_vec))
        return - np.log(y_hat[int(y_true)])

    def loss_mini_batch(self, input_mini_batch, y_true_mini_batch):
        y_hat = softmax_batch(self.feed_forward(input_mini_batch))
        self.batch_loss = - np.sum(np.log(y_hat[np.arange(len(y_hat)), y_true_mini_batch]))
        return self.batch_loss

    def crossover(self, other):
        child = NNModel()
        child2 = NNModel()

        crossover_keep = np.random.binomial(1, 1.0 - 0.5, size=self.w1.shape[1])
        child.w1 = self.w1 * crossover_keep + other.w1 * (1 - crossover_keep)
        child.b1 = self.b1 * crossover_keep + other.b1 * (1 - crossover_keep)
        child2.w1 = other.w1 * crossover_keep + self.w1 * (1 - crossover_keep)
        child2.b1 = other.b1 * crossover_keep + self.b1 * (1 - crossover_keep)

        crossover_keep = np.random.binomial(1, 1.0 - 0.5, size=self.w2.shape[1])
        child.w2 = self.w2 * crossover_keep + other.w2 * (1 - crossover_keep)
        child.b2 = self.b2 * crossover_keep + other.b2 * (1 - crossover_keep)
        child2.w2 = other.w2 * crossover_keep + self.w2 * (1 - crossover_keep)
        child2.b2 = other.b2 * crossover_keep + self.b2 * (1 - crossover_keep)

        crossover_keep = np.random.binomial(1, 1.0 - 0.5, size=self.w3.shape[1])
        child.w3 = self.w3 * crossover_keep + other.w3 * (1 - crossover_keep)
        child.b3 = self.b3 * crossover_keep + other.b3 * (1 - crossover_keep)
        child2.w3 = other.w3 * crossover_keep + self.w3 * (1 - crossover_keep)
        child2.b3 = other.b3 * crossover_keep + self.b3 * (1 - crossover_keep)

        return child, child2

    def mutate(self):
        self.w1 += np.random.normal(0, 0.001, self.w1.shape)
        self.w2 += np.random.normal(0, 0.001, self.w2.shape)
        self.w3 += np.random.normal(0, 0.001, self.w3.shape)

        self.b1 += np.random.normal(0, 0.001, self.b1.shape)
        self.b2 += np.random.normal(0, 0.001, self.b2.shape)
        self.b3 += np.random.normal(0, 0.001, self.b3.shape)

    def loss_and_gradients_mini_batch(self, input_mini_batch, y_true_mini_batch):
        input_dropout_rate = 0.1
        hidden1_dropout_rate = 0.2
        hidden2_dropout_rate = 0.2
        keep1 = 1.0 - hidden1_dropout_rate
        scale1 = 1.0 / keep1
        keep2 = 1.0 - hidden2_dropout_rate
        scale2 = 1.0 / keep2

        input_dropout = np.random.binomial(1, 1.0 - input_dropout_rate, size=input_mini_batch.shape)
        input_layer = input_mini_batch * input_dropout

        out1 = np.dot(input_layer, self.w1) + self.b1
        out1_nl = np.maximum(out1, 0, out1)  # np.tanh(out1)
        out1_dropout = np.random.binomial(1, 1.0 - hidden1_dropout_rate, size=out1_nl.shape) * scale1
        out1_nl_dp = out1_nl * out1_dropout

        out2 = np.dot(out1_nl_dp, self.w2) + self.b2
        out2_nl = np.maximum(out2, 0, out2)  # np.tanh(out2)
        out2_dropout = np.random.binomial(1, 1.0 - hidden2_dropout_rate, size=out2_nl.shape) * scale2
        out2_nl_dp = out2_nl * out2_dropout

        out3 = np.dot(out2_nl_dp, self.w3) + self.b3
        y_hat = softmax_batch(out3)
        loss = - np.sum(np.log(y_hat[np.arange(len(y_hat)), y_true_mini_batch]))

        d_loss_d_out3 = y_hat

        d_loss_d_out3[np.arange(len(d_loss_d_out3)), y_true_mini_batch] -= 1
        g_b3 = np.sum(d_loss_d_out3, axis=0)
        g_w3 = np.einsum('Bi,Bj->ij', out2_nl_dp, d_loss_d_out3)

        d_loss_d_out2_nl_dp = np.dot(self.w3, d_loss_d_out3.T).T
        d_loss_d_out2 = d_loss_d_out2_nl_dp * ((out2 > 0) * 1)   * out2_dropout # * (1 - np.tanh(out2)**2)
        g_b2 = np.sum(d_loss_d_out2, axis=0)
        g_w2 = np.einsum('Bi,Bj->ij', out1_nl, d_loss_d_out2)

        d_loss_d_out1_nl_dp = np.dot(self.w2, d_loss_d_out2.T).T
        d_loss_d_out1 = d_loss_d_out1_nl_dp * ((out1 > 0) * 1)   * out1_dropout # * (1 - np.tanh(out1)**2)
        g_b1 = np.sum(d_loss_d_out1, axis=0)
        g_w1 = np.einsum('Bi,Bj->ij', input_layer, d_loss_d_out1)

        return loss, [g_w1, g_b1, g_w2, g_b2, g_w3, g_b3]

    def train_on_mini_batch(self, input__mini_batch, y_true_mini_batch, learning_rate, regularization):
        loss, grads = self.loss_and_gradients_mini_batch(input__mini_batch, y_true_mini_batch)

        self.w1 -= learning_rate * (grads[0] + self.w1 * regularization)
        self.b1 -= learning_rate * (grads[1] + self.b1 * regularization)
        self.w2 -= learning_rate * (grads[2] + self.w2 * regularization)
        self.b2 -= learning_rate * (grads[3] + self.b2 * regularization)
        self.w3 -= learning_rate * (grads[4] + self.w3 * regularization)
        self.b3 -= learning_rate * (grads[5] + self.b3 * regularization)

        return loss