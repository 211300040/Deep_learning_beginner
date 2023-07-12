import matplotlib.pyplot as plt 
import numpy as np 
def f(x):
    return np.sum(x**2)
x = np.array(np.arange(-3,3,0.1)).reshape(30,2)
print(x)
y = f(x)
print(y)
plt.plot(x,y)
plt.show()
