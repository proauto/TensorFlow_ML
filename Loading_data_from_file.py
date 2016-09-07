import numpy as np

#http://docs scipy.org/doc/numpy-1.10.0/reference/generated/numpy.loadtxt.html

xy = np.loadtxt('train_txt/loading_data_train.txt', unpack=True, dtype='float32')
x_data = xy[0:-1]
y_data = xy[-1]

print('x', x_data)
print('y', y_data)