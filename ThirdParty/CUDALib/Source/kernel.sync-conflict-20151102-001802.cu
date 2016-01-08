#include "kernel.cuh"
#include "kernel_wrapper.h"

//CUDA
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
/*
//Thrust
#include "thrust/host_vector.h"
#include "thrust/device_vector.h"
#include "thrust/copy.h" 
#include "thrust/fill.h"
#include "thrust/sequence.h"
*/
__global__ void InitializeRandom(float* x, int size, float a, float b)
{
  int id = blockIdx.x*blockDim.x + threadIdx.x;
  for (int i = 0; i < size; i++) { x[id+i] = a + b; }
}

void NBodyKernel::Initialize(uint64_t s) {
  size = s;
  x = new float[size];
  cudaMalloc(&d_X, size);
  Initialized = true;
}

void NBodyKernel::CalcNCopy() {
  if (!Initialized) { return; }
  InitializeRandom << <(size + 255) / 256, 256 >> >(d_X, size, 1.0, 2.0);

  cudaMemcpy(d_X, x, size*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(x, d_X, size*sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(d_X);
}
