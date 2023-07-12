import matplotlib.pyplot as plt 
import numpy as np 
x = np.array([0,0,0.2,0.2,0.4,0.4,0.6,0.8,0.8,1.0])
y = np.array([0,0.25,0.25,0.5,0.5,0.75,0.75,0.75,1.0,1.0])
# print(x.shape)
# print(y.shape)
plt.plot(x,y)
plt.ylim(0,1)
plt.xlim(0,1)
plt.draw()
