import numpy as np
import matplotlib.pyplot as plt



data50 = np.load('data50.npy')
data75 = np.load('data75.npy')
data100 = np.load('data100.npy')
data150 = np.load('data150.npy')

plt.figure()
plt.plot(data50, label = '50 hidden nodes')
plt.plot(data75, label = '75 hidden nodes')
plt.plot(data100, label = '100 hidden nodes')
plt.plot(data150, label = '150 hidden nodes')

plt.xlabel('Number of epochs')
plt.ylabel('Mean-Squared Error')
plt.title('Normal Scale')
plt.legend() 

plt.figure()
plt.loglog(data50, label = '50 hidden nodes')
plt.loglog(data75, label = '75 hidden nodes')
plt.loglog(data100, label = '100 hidden nodes')
plt.loglog(data150, label = '150 hidden nodes')


plt.xlabel('Number of epochs')
plt.ylabel('Mean-Squared Error')
plt.title('Loglog scale')
plt.legend() 

plt.show()