from keras.layers import Input, Dense
from keras.models import Model
from keras import optimizers

import numpy as np
import matplotlib.pyplot as plt


loaddata = False
if loaddata:
	train_input = np.genfromtxt('datasets/bindigit_trn.csv',dtype ='int', delimiter=',')
	np.save('bindigit_trn_npy',train_input)
else:
	train_input = np.load('bindigit_trn_npy.npy')

#train_target = readData('targetdigit_trn')

####### Parameters ########
# this is the size of our encoded representations
encoding_dim = 50 # 32 floats -> compression of factor 24.5, assuming the input is 784 floats
nb_epochs = 1


# this is our input placeholder
input_img = Input(shape=(784,))
# "encoded" is the encoded representation of the input
encoded = Dense(encoding_dim, activation='relu',kernel_initializer='random_normal')(input_img)
# "decoded" is the lossy reconstruction of the input
decoded = Dense(784, activation='relu',kernel_initializer='random_normal')(encoded)

# this model maps an input to its reconstruction
autoencoder = Model(input_img, decoded)
#training of the model
sgd = optimizers.SGD(lr=0.5, momentum = 0.2)
autoencoder.compile(optimizer=sgd, loss='mean_squared_error')
# autoencoder.compile(optimizer='adam', loss='mean_squared_error')
hist = autoencoder.fit(train_input, train_input,
                epochs=nb_epochs,
                batch_size=256,
                shuffle=True, verbose=1)

#plotting of the error
# plt.figure()
# plt.plot(hist.history['loss'])
# plt.xlabel('Number of epochs')
# plt.ylabel('Mean-Squared Error')
# plt.show()
# plt.figure()
# plt.loglog(hist.history['loss'])
# plt.xlabel('Number of epochs')
# plt.ylabel('Mean-Squared Error')

# np.save('data150',hist.history['loss'])
# #decoded images
# decoded_imgs = autoencoder.predict(train_input)

# set_images = [11,2,8,17,21,10,0,4,31,9]
# plt.figure(figsize=(20, 4))
# for i in range(10):
#     # original
#     j=set_images[i]
#     plt.subplot(2, 10, i + 1)
#     plt.imshow(train_input[j].reshape(28, 28))
#     plt.gray()
#     plt.axis('off')
 
#     # reconstruction
#     plt.subplot(2, 10, i + 1 + 10)
#     plt.imshow(decoded_imgs[j].reshape(28, 28))
#     plt.gray()
#     plt.axis('off')
 
# plt.tight_layout()
# plt.show()

#### show the weigths
weights = autoencoder.get_weights()[2]
plt.figure(figsize=(20,20))
for i in range(encoding_dim):
	plt.subplot(2, 10, i + 1)
	plt.imshow(weights[i].reshape(28, 28))
	plt.gray()
	plt.axis('off')
plt.tight_layout()
plt.show()