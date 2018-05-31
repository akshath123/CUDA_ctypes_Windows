#include <iostream>
#include <cuda.h>

using namespace std ;

# define DELLEXPORT extern "C" __declspec(dllexport)

__global__ void cudaSquareKernel(float * d_in, float * d_out){
	
	int idx = blockIdx.x * blockIdx.x + threadIdx.x ;
	d_out[idx] = d_in[idx] * d_in[idx] ;
}

DELLEXPORT void cudaSquare(float * h_in, float * h_out, int arr_size){
	
	const long long int ARRAY_BYTES = arr_size * sizeof(float) ;

	float *d_in, *d_out ;

	cudaMalloc((void **) &d_in, ARRAY_BYTES) ;
	cudaMalloc((void **) &d_out, ARRAY_BYTES) ;

	cudaMemcpy(d_in, h_in, ARRAY_BYTES, cudaMemcpyHostToDevice) ;
	
	cudaSquareKernel<<< 1, arr_size >>>(d_in, d_out) ;

	cudaMemcpy(h_out, d_out, ARRAY_BYTES, cudaMemcpyDeviceToHost) ;

	cudaFree(d_in) ;
	cudaFree(d_out) ;
}
