import numpy as np 
import time 
a = np.random.rand(10000000)
b = np.random.rand(10000000)
# print(a)
tic = time.time()
# c = np.dot(a,b)
c = a @ b
toc = time.time()
print("Vectorized version:"+ str(1000*(toc-tic)) + "ms")
# 在numpy中，用一个列向量和一个行向量相加：会得到一个矩阵（广播机制）
a = np.array([1,2,3,4,5]).reshape(1,-1)
b = np.array([-1,-2,-3,-4,-5]).reshape(-1,1)
# print(a,"+",b)
print(a.shape,'+',b.shape)
print(a+b)
