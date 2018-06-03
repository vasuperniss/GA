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
import operator


def train_classifier(train_set, dev_set, num_iterations, models, roulette, elitism=8):
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
            population.sort(key=operator.attrgetter('batch_loss'))

            avg_loss += population[0].batch_loss
            if batch_index % 1000 == 0 and batch_index != 0:
                e_time = time.time()
                print '\t', batch_index / float(length), '% complete. took:', e_time - s_time,\
                    'average loss <-', avg_loss / 1000
                avg_loss = 0
                s_time = time.time()
            if batch_index % 5000 == 0 and batch_index != 0:
                print '\taccuracy on dev: best:', accuracy_on_dataset(dev_set, population[0]),\
                    'loss of best:', '%.2f' % (population[0].batch_loss / batch_size), \
                    '\n\taccuracy on dev: worst:', accuracy_on_dataset(dev_set, population[-1]), \
                    'loss of worst:', '%.2f' % (population[-1].batch_loss / batch_size)

            new_population = []
            for e in range(elitism):
                new_population.append(population[e][1])

            for c in range(0, population_size - elitism, 2):
                p1_index = rand.choice(roulette)
                p2_index = rand.choice(roulette)

                parent1 = population[p1_index][1]
                parent2 = population[p2_index][1]

                child1, child2 = parent1.crossover(parent2)
                if rand.randint(0,2) == 1:
                    child1.mutate()
                if rand.randint(0,2) == 1:
                    child2.mutate()

                new_population.append(child1)
                new_population.append(child2)

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
            model.initialize(28 * 28, 256, 128, 10)
            models.append(model)

    train_set, dev_set = load_mnist('../mnist_data')

    roulette_wheel = []
    for i in range(population_size):
        for k in range(int((population_size - i) ** 0.5)):
            roulette_wheel.append(i)
    for i in range(7):
        rand.shuffle(roulette_wheel)

    train_classifier(train_set, dev_set, 30, models, roulette_wheel)
