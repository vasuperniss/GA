# ###############################
# Michael Vassernis  -  319582888
#
#################################
import ga_model as ga_mdl
import numpy as np
import random as rand
import time
from helper_functions import load_mnist, accuracy_on_dataset


def calc_loss(models, batch):
    for m in models:
        m.loss_on_batch(batch)

def crossovers(population, losses, roulette, new_population, blank_models, chieldren_count):
    for i in range(chieldren_count):
        parent1 = losses[roulette[rand.randint(0, len(roulette_wheel) - 1)]][1]
        parent2 = losses[roulette[rand.randint(0, len(roulette_wheel) - 1)]][1]
        blank = blank_models.pop()
        child = parent1.mate_with(parent2, blank)
        new_population.append(child)


def train_classifier(train_set, dev_set, num_iterations, learning_rate, models, roulette):
    l = len(train_set)
    population_size = len(models)
    blank_models = []
    for i in range(population_size - 2):
        blank = ga_mdl.Genetic_NNModel([28*28, 200, 10])
        blank_models.append(blank)

    batch_size = 50
    population = models
    for I in xrange(num_iterations):
        np.random.shuffle(train_set)
        avg_loss = 0
        print '\nbegan doing epoch no.', I + 1
        s_time = time.time()
        for k in range(0, l, batch_size):
            losses = []
            batch = train_set[k:k + batch_size]
            calc_loss(population, batch)
            for model in population:
                losses.append([model.batch_loss, model])
            losses = sorted(losses)
            avg_loss += losses[0][0]
            if k % 1000 == 0 and k != 0:
                e_time = time.time()
                print '\t', k / float(l), '% complete. took:', e_time - s_time, 'average loss <-', avg_loss / 1000
                avg_loss = 0
                s_time = time.time()
            if k % 5000 == 0:
                print '\taccuracy on dev:', accuracy_on_dataset(dev_set, losses[0][1])
            new_population = []
            new_population.append(losses[0][1])
            new_population.append(losses[1][1])
            crossovers(population, losses, roulette, new_population, blank_models, population_size - 2)
            for i in range(2, population_size, 1):
                blank_models.append(losses[i][1])
            population = new_population
        train_accuracy = accuracy_on_dataset(train_set, model)
        dev_accuracy = accuracy_on_dataset(dev_set, model)
        print '****** EPOCH SUMMARY', I, train_accuracy, dev_accuracy


if __name__ == '__main__':
    models = []
    population_size = 50
    for i in range(population_size):
        model = ga_mdl.Genetic_NNModel([28*28, 200, 10])
        models.append(model)

    train_set, dev_set = load_mnist('../mnist_data')

    roulette_wheel = []
    for i in range(population_size):
        for k in range(int((population_size - i) ** 1.1)):
            roulette_wheel.append(i)
    for i in range(7):
        rand.shuffle(roulette_wheel)

    train_classifier(train_set, dev_set, 20, 0.01, models, roulette_wheel)
