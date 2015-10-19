#include <stdio.h>

#include "kernel_wrapper.h"

//CUDA
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//Thrust
#include "thrust/host_vector.h"
#include "thrust/device_vector.h"
#include "thrust/copy.h" 
#include "thrust/fill.h"
#include "thrust/sequence.h"

__global__ void InitializeRandom(float* x, int size)
{
  int id = blockIdx.x*blockDim.x + threadIdx.x;
  for (int i = 0; i < size; i++) {
    x[id+i] = 1;
  }
}

float kernel(int size)
{
  int bSize = size*sizeof(float);
  float* x; x = (float*)malloc(bSize);
  float* d_X; cudaMalloc(&d_X, bSize);
  InitializeRandom << <(size+255)/256, 256 >> >(d_X, size);

  cudaMemcpy(x, d_X, bSize, cudaMemcpyDeviceToHost);
  cudaFree(d_X);
  
  float r = 0;
  for (int i = 0; i < size; i++) { r += x[i]; }
  
  delete x;
  return r;
}