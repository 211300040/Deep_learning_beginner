import numpy as np 
a = np.array([True,False])
# print(a)
a = a.astype('float64')
# print(a)
a = np.array([[1,2,3],
              [4,5,6]])
a = np.array([1,2,3,4,5]).reshape(-1,1)
print(a)
print(np.sum(a,axis=1,keepdims=True))
