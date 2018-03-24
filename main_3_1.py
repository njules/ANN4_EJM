import numpy as np
from sklearn.neural_network import BernoulliRBM
import csv
import matplotlib.cm as cm
import matplotlib.pyplot as plt

def readData(file_name):
    data_matrix = []
    with open('datasets/' + file_name + '.csv') as csv_file:
        csv_reader = csv.reader(csv_file, quoting=csv.QUOTE_NONNUMERIC)
        for data_vector in csv_reader:
            data_matrix += [data_vector]
    return np.array(data_matrix)

train_input = readData('bindigit_trn')
train_target = readData('targetdigit_trn')

plt.imsave('test.png', train_input[0].reshape(28,28), cmap=cm.gray)

rbm = BernoulliRBM(n_components=50, learning_rate=.01, batch_size=100, n_iter=20)
rbm.fit(train_input)