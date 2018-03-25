import numpy as np
from sklearn.neural_network import BernoulliRBM
import csv
import matplotlib.cm as cm
import matplotlib.pyplot as plt

first_number_index = [11, 2, 8, 15, 6, 10, 0, 4, 31, 9]

def readData(file_name):
    data_matrix = []
    with open('datasets/' + file_name + '.csv') as csv_file:
        csv_reader = csv.reader(csv_file, quoting=csv.QUOTE_NONNUMERIC)
        for data_vector in csv_reader:
            data_matrix += [data_vector]
    return np.array(data_matrix)

train_input = readData('bindigit_trn')
train_target = readData('targetdigit_trn')
test_input = readData('bindigit_tst')
test_target = readData('targetdigit_tst')

rbm = BernoulliRBM(n_components=150, learning_rate=.2, batch_size=100, n_iter=50)
rbm.fit(train_input, y=test_input)

original = []
for image in range(len(first_number_index)):
    original += [train_input[first_number_index[image]]]

reconstructed_boolean = rbm.gibbs(original)
reconstructed = np.zeros(np.shape(reconstructed_boolean))
for i in range(np.shape(reconstructed_boolean)[0]):
    for j in range(np.shape(reconstructed_boolean)[1]):
        if reconstructed_boolean[i][j]:
            reconstructed[i][j] = 1
        else:
            reconstructed[i][j] = 0
for idx in range(10):
    plt.imsave('images/rbm_org_'+str(idx)+'.png', original[idx].reshape(28, 28), cmap=cm.gray)
    plt.imsave('images/rbm_rec_'+str(idx)+'.png', reconstructed[idx].reshape(28, 28), cmap=cm.gray)