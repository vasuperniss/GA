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


def train_classifier(train_set, dev_set, num_iterations, learning_rate, model, regularization, model_file):
    best_dev_accuracy = 0.0
    for epoch in xrange(num_iterations):
        np.random.shuffle(train_set)
        print '\nbegan doing epoch no.', epoch + 1
        total_loss = 0.0  # total loss in this iteration.
        avg_loss = 0.0
        count = 0.0
        batch_size = 10
        start_time = time.time()
        for batch_index in range(0, len(train_set), batch_size):
            loss = model.train_on_mini_batch(train_set[batch_index:batch_index+batch_size,:-1],
                                            train_set[batch_index:batch_index+batch_size,-1].astype(dtype=int),
                                            learning_rate, regularization)
            total_loss += loss
            avg_loss += loss
            count += batch_size
            if count % 5000 == 0:
                took = time.time() - start_time
                start_time = time.time()
                print '\t', (count / len(train_set)) * 100, '% complete.', 'took:', took,\
                    'seconds. average loss <-', avg_loss / 5000
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
    model_file = '../saved_models/nn2classifier.mdl'
    train_set, dev_set = load_mnist('../mnist_data')

    if len(sys.argv) > 1:
        model = load_model(model_file)
    else:
        model = nn_mdl.NNModel()
        model.initialize(28*28, 128, 128, 10)

    train_classifier(train_set, dev_set, num_iterations=20, learning_rate=0.001, model=model, regularization=1e-6,
                     model_file=model_file)
