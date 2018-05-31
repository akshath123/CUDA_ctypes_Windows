import numpy as np
import ctypes
from ctypes import * 

def get_cuda_square():
	dll = ctypes.windll.LoadLibrary("H:\Cuda Programming\CUDA Programs\squareGPU.dll") 
	func = dll.cudaSquare
	func.argtypes = [POINTER(c_float), POINTER(c_float), c_size_t] 
	return func

__cuda_square = get_cuda_square()

def cuda_square(a, b, size):
	a_p = a.ctypes.data_as(POINTER(c_float))
	b_p = b.ctypes.data_as(POINTER(c_float))

	__cuda_square(a_p, b_p, size)

if __name__ == '__main__':
	size = int(1024) 

	a = np.arange(1, size + 1).astype('float32')
	b = np.zeros(size).astype('float32')

	cuda_square(a, b, size)

	for i in range(size):
		print(b[i], end = "")
		print( '\t' if ((i % 4) != 3) else "\n", end = " ", flush = True)
