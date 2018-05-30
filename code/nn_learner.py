# ###############################
# Michael Vassernis  -  319582888
#
#################################
import nn_model as nn_mdl
from helper_functions import load_mnist, accuracy_on_dataset, load_model, save_model
import numpy as np
import matplotlib.pyplot as plt
import sys
import time


def train_auto_encoder(train_set, dev_set, num_iterations, learning_rate, model, regularization, model_file):
    for epoch in xrange(num_iterations):
        np.random.shuffle(train_set)
        print '\nbegan doing epoch no.', epoch + 1
        total_loss = 0.0  # total loss in this iteration.
        avg_loss = 0.0
        count = 0.0
        start_time = time.time()
        for example in train_set:
            loss = model.train_on_example_sqrt(example[:-1], example[:-1], learning_rate, regularization)
            total_loss += loss
            avg_loss += loss
            count += 1
            if count % 5000 == 0:
                took = time.time() - start_time
                start_time = time.time()
                print '\t', (count / len(train_set)) * 100, '% complete.', 'took:', took,\
                    'seconds. average loss <-', avg_loss / 5000
                model.add_loss_data(avg_loss)
                avg_loss = 0
        train_loss = total_loss / len(train_set)
        print 'epoch:', epoch + 1, 'loss:', train_loss
        print 'saving model....'
        save_model(model, model_file)


def train_classifier(train_set, dev_set, num_iterations, learning_rate, model, regularization, model_file):
    best_dev_accuracy = 0.0
    for epoch in xrange(num_iterations):
        np.random.shuffle(train_set)
        print '\nbegan doing epoch no.', epoch + 1
        total_loss = 0.0  # total loss in this iteration.
        avg_loss = 0.0
        count = 0.0
        start_time = time.time()
        for example in train_set:
            loss = model.train_on_example(example[:-1], example[-1], learning_rate, regularization)
            total_loss += loss
            avg_loss += loss
            count += 1
            if count % 5000 == 0:
                took = time.time() - start_time
                start_time = time.time()
                print '\t', (count / len(train_set)) * 100, '% complete.', 'took:', took,\
                    'seconds. average loss <-', avg_loss / 5000
                model.add_loss_data(avg_loss)
                avg_loss = 0
        train_loss = total_loss / len(train_set)
        train_accuracy = accuracy_on_dataset(train_set, model)
        dev_accuracy = accuracy_on_dataset(dev_set, model)
        print 'epoch:', epoch + 1, 'loss:', train_loss, 'train accuracy:', train_accuracy, 'dev accuracy:', dev_accuracy
        if dev_accuracy > best_dev_accuracy:
            best_dev_accuracy = dev_accuracy
            print 'saving model....'
            save_model(model, model_file)


if __name__ == '__main__':
    model_file = '../saved_models/nn1classifier.mdl.pickle'
    if len(sys.argv) > 1:
        model = load_model(model_file)
        # print model.loss_data
    else:
        model = nn_mdl.NNModel([28*28, 256, 10])

    train_set, dev_set = load_mnist('../mnist_data')


    # plt.imshow(np.concatenate((np.concatenate((train_set[0][0:-1].reshape((28,28)), model.feed_forward(train_set[0][0:-1]).reshape((28,28))), axis=1),
    #                            np.concatenate((train_set[1][0:-1].reshape((28, 28)), model.feed_forward(train_set[1][0:-1]).reshape((28, 28))), axis=1),
    #                            np.concatenate((train_set[2][0:-1].reshape((28, 28)), model.feed_forward(train_set[2][0:-1]).reshape((28, 28))), axis=1),
    #                            np.concatenate((train_set[3][0:-1].reshape((28, 28)), model.feed_forward(train_set[3][0:-1]).reshape((28, 28))), axis=1),
    #                            np.concatenate((train_set[4][0:-1].reshape((28, 28)), model.feed_forward(train_set[4][0:-1]).reshape((28, 28))), axis=1),
    #                            np.concatenate((train_set[5][0:-1].reshape((28, 28)), model.feed_forward(train_set[5][0:-1]).reshape((28, 28))), axis=1),),
    #                           axis=0))
    # plt.show()

    # model_file = '../saved_models/nn2classifier.mdl.pickle'
    # model = nn_mdl.NNModel([256, 256, 10])
    # model = nn_mdl.NNModel([256, 200, 200, 10])
    # model = load_model(model_file)
    train_classifier(train_set, dev_set, num_iterations=20, learning_rate=0.0001, model=model, regularization=1e-7, model_file=model_file)

    # train_auto_encoder(train_set, dev_set, num_iterations=20, learning_rate=0.001, model=model,
