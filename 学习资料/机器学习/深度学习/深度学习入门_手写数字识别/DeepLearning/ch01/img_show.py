# coding: utf-8
import matplotlib.pyplot as plt
import sys,os 
from matplotlib.image import imread
print(os.pardir)
sys.path.append(os.pardir)
print(sys.path)
img = imread('dataset/lena.png') # 读入图像
plt.imshow(img)

plt.show()
