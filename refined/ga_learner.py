# ###############################
# Michael Vassernis  -  319582888
#
#################################
import nn_model as nn_mdl
import numpy as np
import random as rand
import time
from helper_functions import load_mnist, accuracy_on_dataset, save_model, load_model
import sys


def train_classifier(train_set, dev_set, num_iterations, models, roulette, elitism=4):
    length = len(train_set)
    population_size = len(models)

    batch_size = 100
    population = models

    for epoch in range(num_iterations):
        np.random.shuffle(train_set)
        avg_loss = 0
        print '\nbegan doing epoch no.', epoch + 1
        s_time = time.time()
        for batch_index in range(0, length, batch_size):
            batch_x = train_set[batch_index:batch_index + batch_size, :-1]
            batch_y = train_set[batch_index:batch_index + batch_size, -1].astype(dtype=int)
            for m in population:
                m.loss_mini_batch(batch_x, batch_y)

            losses = []
            for model in population:
                losses.append([model.batch_loss, model])
            losses = sorted(losses)

            avg_loss += losses[0][0]
            if batch_index % 1000 == 0 and batch_index != 0:
                e_time = time.time()
                print '\t', batch_index / float(length), '% complete. took:', e_time - s_time,\
                    'average loss <-', avg_loss / 1000
                avg_loss = 0
                s_time = time.time()
            if batch_index % 5000 == 0 and batch_index != 0:
                print '\taccuracy on dev: best:', accuracy_on_dataset(dev_set, losses[0][1]),\
                    'loss of best:', '%.2f' % (losses[0][0] / batch_size), \
                    '\n\taccuracy on dev: worst:', accuracy_on_dataset(dev_set, losses[-1][1]), \
                    'loss of worst:', '%.2f' % (losses[-1][0] / batch_size)

            new_population = []
            for e in range(elitism):
                new_population.append(losses[e][1])

            for c in range(population_size - elitism):
                p1_index = rand.choice(roulette)
                p2_index = rand.choice(roulette)

                parent1 = losses[p1_index][1]
                parent2 = losses[p2_index][1]

                child = parent1.crossover(parent2)
                child.mutate()

                new_population.append(child)

            population = new_population
        train_accuracy = accuracy_on_dataset(train_set, population[0])
        dev_accuracy = accuracy_on_dataset(dev_set, population[0])
        print '****** EPOCH SUMMARY', epoch + 1, train_accuracy, dev_accuracy
        print '****** saving population...'
        count = 0
        for m in population:
            save_model(m, '../saved_models/ga200' + str(count) + '.mdl')
            count += 1


if __name__ == '__main__':
    models = []
    population_size = 50
    if len(sys.argv) > 1:
        for i in range(population_size):
            model = load_model('../saved_models/ga200' + str(i) + '.mdl')
            models.append(model)
    else:
        for i in range(population_size):
            model = nn_mdl.NNModel()
            model.initialize(28 * 28, 200, 200, 10)
            models.append(model)

    train_set, dev_set = load_mnist('../mnist_data')

    roulette_wheel = []
    for i in range(population_size):
        for k in range(int((population_size - i) ** 1)):
            roulette_wheel.append(i)
    for i in range(7):
        rand.shuffle(roulette_wheel)

    train_classifier(train_set, dev_set, 30, models, roulette_wheel)
