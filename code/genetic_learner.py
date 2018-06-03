# ###############################
# Michael Vassernis  -  319582888
#
#################################
import ga_model as ga_mdl
import numpy as np
import random as rand
import time
from helper_functions import load_mnist, accuracy_on_dataset, save_model, load_model
import sys


def calc_loss_model(model_batch):
    return model_batch[0].loss_on_batch(model_batch[1])


def calc_loss(models, batch):
    for m in models:
        m.loss_on_batch2(batch)


def calc_loss_with_pool(models, batch, pool):
    map_list = []
    for m in models:
        map_list.append((m, batch))
    pool.map(calc_loss_model, map_list)
    # pool.join()


def crossovers(losses, roulette, new_population, blank_models, children_count):
    for c in range(children_count):
        p1_index = roulette[rand.randint(0, len(roulette_wheel) - 1)]
        p2_index = roulette[rand.randint(0, len(roulette_wheel) - 1)]
        parent1 = losses[p1_index][1]
        parent2 = losses[p2_index][1]
        blank = blank_models.pop()
        child = parent1.mate_with(parent2, blank, int((p1_index + p2_index) / 2))
        new_population.append(child)


def train_classifier(train_set, dev_set, num_iterations, models, roulette):
    length = len(train_set)
    population_size = len(models)
    blank_models = []
    for b in range(population_size - 4):
        blank = ga_mdl.Genetic_NNModel([28*28, 128, 10])
        blank_models.append(blank)

    batch_size = 200
    population = models
    best = None
    for epoch in range(num_iterations):
        np.random.shuffle(train_set)
        avg_loss = 0
        print '\nbegan doing epoch no.', epoch + 1
        s_time = time.time()
        for batch_index in range(0, length, batch_size):
            losses = []
            batch = train_set[batch_index:batch_index + batch_size]

            # s_t = time.time()
            calc_loss(population, batch)
            # print time.time() - s_t
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
            if batch_index % 5000 == 0:
                print '\taccuracy on dev: best:', accuracy_on_dataset(dev_set, losses[0][1]),\
                    'loss of best:', '%.2f' % (losses[0][0] / batch_size), \
                    '\n\taccuracy on dev: worst:', accuracy_on_dataset(dev_set, losses[-1][1]), \
                    'loss of worst:', '%.2f' % (losses[-1][0] / batch_size)
            new_population = []
            best = losses[0][1]
            new_population.append(losses[0][1])
            new_population.append(losses[1][1])
            new_population.append(losses[2][1])
            new_population.append(losses[3][1])
            crossovers(losses, roulette, new_population, blank_models, population_size - 4)
            blank_models = []
            for j in range(2, population_size, 1):
                blank_models.append(losses[j][1])
            population = new_population
        train_accuracy = accuracy_on_dataset(train_set, best)
        dev_accuracy = accuracy_on_dataset(dev_set, best)
        print '****** EPOCH SUMMARY', epoch + 1, train_accuracy, dev_accuracy
        print '****** saving population...'
        count = 0
        for m in population:
            save_model(m, '../saved_models/ga200_p2after_m' + str(count) + '.mdl')
            count += 1


if __name__ == '__main__':
    models = []
    population_size = 100
    if len(sys.argv) > 1:
        for i in range(population_size):
            if i < population_size:
                model = load_model('../saved_models/ga200_p2after_m' + str(i) + '.mdl')
            else:
                model = ga_mdl.Genetic_NNModel([28 * 28, 128, 10])
            models.append(model)
    else:
        for i in range(population_size):
            model = ga_mdl.Genetic_NNModel([28*28, 128, 10])
            models.append(model)

    train_set, dev_set = load_mnist('../mnist_data')

    roulette_wheel = []
    for i in range(population_size):
        for k in range(int((population_size - i) ** 0.5)):
            roulette_wheel.append(i)
    for i in range(7):
        rand.shuffle(roulette_wheel)

    train_classifier(train_set, dev_set, 20, models, roulette_wheel)
