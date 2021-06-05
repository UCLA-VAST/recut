import numpy as np

def reorder(arr, order=[0,2,6,8,1,7,3,5,4]):
	arr = np.moveaxis(arr, [1, 2, 3], [2, 3, 1])
	arr = arr.reshape((8192, 9))
	arr = np.array([[arr[j][i] for i in order] for j in range(len(arr))])
	arr = arr.reshape((128, 64, 3, 3))
	return np.moveaxis(arr, [1, 2, 3], [3, 1, 2])