import nn_model as mdl
import numpy as np
import struct
import random as rand
import time
from threading import Thread


def read_idx(filename):
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.fromstring(f.read(), dtype=np.uint8).reshape(shape)


def accuracy_on_dataset(data_X, data_Y, model):
    good = bad = 0.0
    l = len(data_Y)
    for i in range(l):
        if model.predict(data_X[i]) == data_Y[i]:
            good += 1
        else:
            bad += 1
    return good / (good + bad)


def loss_on_dataset(data_X, data_Y, model):
    l = len(data_Y)
    loss = 0.0
    for i in range(l):
        loss += model.loss(data_X[i], data_Y[i])
    return loss


def calc_loss(models, batch_X, batch_Y):
    for m in models:
        m.loss_on_batch(batch_X, batch_Y)

def crossovers(population, losses, roulette, new_population, blank_models, chieldren_count):
    for i in range(chieldren_count):
        parent1 = losses[roulette[rand.randint(0, len(roulette_wheel) - 1)]][1]
        parent2 = losses[roulette[rand.randint(0, len(roulette_wheel) - 1)]][1]
        if parent1 == parent2:
            parent2 = losses[roulette[rand.randint(0, len(roulette_wheel) - 1)]][1]
        if parent1 == parent2:
            parent2 = losses[roulette[rand.randint(0, len(roulette_wheel) - 1)]][1]
        if parent1 == parent2:
            parent2 = losses[roulette[rand.randint(0, len(roulette_wheel) - 1)]][1]
        if parent1 == parent2:
            parent2 = losses[roulette[rand.randint(0, len(roulette_wheel) - 1)]][1]
        blank = blank_models.pop()
        child = parent1.mate_with(parent2, blank)
        new_population.append(child)


def train_classifier(train_X, train_Y, dev_X, dev_Y, num_iterations, learning_rate, models, roulette):
    l = len(train_Y)
    population_size = len(models)
    blank_models = []
    # blank_models2 = []
    for i in range(population_size - 2):
        blank = mdl.NNModel_Reversed([28*28, 200, 10])
        blank_models.append(blank)
        # blank = mdl.NNModel_Reversed([28 * 28, 200, 10])
        # blank_models2.append(blank)

    batch_size = 50
    population = models
    avg_loss = 0.0
    for I in xrange(num_iterations):
        avg_loss = 0
        print '\nbegan doing epoch no.', I + 1
        s_time = time.time()
        for k in range(0, l, batch_size):
            losses = []
            batch_X = train_X[k:k + batch_size]
            batch_Y = train_Y[k:k + batch_size]
            # t1 = Thread(target=calc_loss, args=(population[0:population_size / 2], batch_X, batch_Y))
            # t2 = Thread(target=calc_loss, args=(population[population_size / 2:population_size], batch_X, batch_Y))
            # t1.start()
            calc_loss(population, batch_X, batch_Y)
            # t2.start()
            # t1.join()
            # t2.join()
            for model in population:
                losses.append([model.batch_loss, model])
                # losses.append([loss_on_dataset(train_X[0:0 + batch_size], train_Y[0:0 + batch_size], model), model])
            losses = sorted(losses)
            # print 'average loss of best model:', losses[0][0] / batch_size
            # print losses
            avg_loss += losses[0][0]
            if k % 1000 == 0 and k != 0:
                e_time = time.time()
                print '\t', k / float(l), '% complete. took:', e_time - s_time, 'average loss <-', avg_loss / 1000
                avg_loss = 0
                s_time = time.time()
            if k % 5000 == 0:
                # print '\taccuracy on batch:', accuracy_on_dataset(train_X[0:0 + batch_size], train_Y[0:0 + batch_size], losses[0][1])
                print '\taccuracy on dev:', accuracy_on_dataset(dev_X, dev_Y, losses[0][1])
            new_population = []
            new_population.append(losses[0][1])
            new_population.append(losses[1][1])
            # for i in range(population_size - 1):
            #     parent1 = losses[roulette[rand.randint(0, len(roulette_wheel) - 1)]][1]
            #     parent2 = losses[roulette[rand.randint(0, len(roulette_wheel) - 1)]][1]
            #     if parent1 == parent2:
            #         parent2 = losses[roulette[rand.randint(0, len(roulette_wheel) - 1)]][1]
            #     if parent1 == parent2:
            #         parent2 = losses[roulette[rand.randint(0, len(roulette_wheel) - 1)]][1]
            #     if parent1 == parent2:
            #         parent2 = losses[roulette[rand.randint(0, len(roulette_wheel) - 1)]][1]
            #     if parent1 == parent2:
            #         parent2 = losses[roulette[rand.randint(0, len(roulette_wheel) - 1)]][1]
            #     blank = blank_models.pop()
            #     child = parent1.mate_with(parent2, blank)
            #     new_population.append(child)
            # for i in range(population_size - 1):
            #     blank_models.append(losses.pop()[1])
            # new_pop = []
            # new_pop2 = []
            # crossovers(population, losses, roulette, new_pop1, blank_models1, population_size / 2)
            # t1 = Thread(target=crossovers, args=(population, losses, roulette, new_pop1, blank_models1, population_size / 2))
            # t2 = Thread(target=crossovers, args=(population, losses, roulette, new_pop2, blank_models2, population_size / 2 - 1))
            # t1.start()
            crossovers(population, losses, roulette, new_population, blank_models, population_size - 2)
            for i in range(2, population_size, 1):
                blank_models.append(losses[i][1])
            # t2.start()
            # t1.join()
            # t2.join()
            # for m in new_pop:
            #     new_population.append(m)

            # for m in new_pop2:
            #     new_population.append(m)
            # blank_models1 = population[0:population_size / 2]
            # blank_models2 = population[population_size / 2:population_size - 1]
            # blank_models1 = []
            # blank_models2 = []
            # for i in range(population_size):
            #     blank = mdl.NNModel_Reversed([28 * 28, 200, 10])
            #     blank_models1.append(blank)
            #     blank = mdl.NNModel_Reversed([28 * 28, 200, 10])
            #     blank_models2.append(blank)
            population = new_population
        train_accuracy = accuracy_on_dataset(train_X, train_Y, model)
        dev_accuracy = accuracy_on_dataset(dev_X, dev_Y, model)
        print '****** EPOCH SUMMARY', I, train_accuracy, dev_accuracy


if __name__ == '__main__':
    models = []
    population_size = 50
    for i in range(population_size):
        model = mdl.NNModel_Reversed([28*28, 200, 10])
        models.append(model)

    train_Y = read_idx('mnist_data/train-labels-idx1-ubyte/data')
    train_X = read_idx('mnist_data/train-images-idx3-ubyte/data')
    train_X = train_X.reshape((len(train_Y), 28*28)) / 255.0

    dev_X = train_X[50000:60000]
    dev_Y = train_Y[50000:60000]

    train_X = train_X[0:50000]
    train_Y = train_Y[0:50000]

    roulette_wheel = []
    for i in range(population_size):
        for k in range(int((population_size - i) ** 1.1)):
            roulette_wheel.append(i)
    for i in range(7):
        rand.shuffle(roulette_wheel)
    # print roulette_wheel

    # pari_X = np.zeros((20000, 30))
    # pari_Y = np.zeros(20000, dtype=int)
    #
    # pari_d_X = np.zeros((5000, 30))
    # pari_d_Y = np.zeros(5000, dtype=int)
    #
    # for i in range(20000):
    #     a = np.random.choice([0, 1], size=(30,), p=[1./2, 1./2])
    #     res = np.sum(a)
    #     if res % 2 == 0:
    #         res = 1
    #     else:
    #         res = 0
    #     pari_X[i] = a
    #     pari_Y[i] = res
    #
    # for i in range(5000):
    #     a = np.random.choice([0, 1], size=(30,), p=[1./2, 1./2])
    #     res = np.sum(a)
    #     if res % 2 == 0:
    #         res = 1
    #     else:
    #         res = 0
    #     pari_d_X[i] = a
    #     pari_d_Y[i] = res
    #
    # train_classifier(pari_X, pari_Y, pari_d_X, pari_d_Y, 20, 0.01, models, roulette_wheel)
    train_classifier(train_X, train_Y, dev_X, dev_Y, 20, 0.01, models, roulette_wheel)
