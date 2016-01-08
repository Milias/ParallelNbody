#include "kernel.cuh"
#include "kernel_wrapper.h"

#ifndef __CUDACC__
  #define __CUDACC__
#endif

//CUDA
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_functions.h>

//Thrust
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/fill.h>
#include <thrust/copy.h>
#include <thrust/remove.h>
#include <thrust/execution_policy.h>
#include <thrust/count.h>

#define THREADS_PER_BLOCK 512
#define PARTICLES_PER_LEAF 1

__device__ uint64_t SplitBy3(uint64_t x)
{
  x &= 0x1fffff;
  x = (x | (x << 32)) & 0x1f00000000ffff;
  x = (x | (x << 16)) & 0x1f0000ff0000ff;
  x = (x | (x << 8)) & 0x100f00f00f00f00f;
  x = (x | (x << 4)) & 0x10c30c30c30c30c3;
  x = (x | (x << 2)) & 0x1249249249249249;
  return x;
}

__device__ uint64_t MagicBits(uint64_t x, uint64_t y, uint64_t z)
{
  return SplitBy3(x) | SplitBy3(y) << 1 | SplitBy3(z) << 2;
}

__device__ uint64_t KerEncode(float4 v)
{
  return MagicBits((uint64_t)(v.x), (uint64_t)(v.y), (uint64_t)(v.z));
}

__global__ void ParticleAlloc(uint32_t N, float4 *Data, KerParticle *Particles, KerNode *Leaves)
{
  uint32_t i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < N) {
    Particles[i].Position = Data[i];
    Leaves[i].Particle = Particles + i;
    Particles[i].ParentNode = Leaves + i;
    Particles[i].Encoded = KerEncode(Particles[i].Position);
  }
}

__device__ uint2 CalculateRange(uint32_t N, KerParticle* Particles, uint32_t index)
{
  if (index == 0) { return make_uint2(0, N); }
  int dir;
  uint32_t d_min, iind;
  iind = index;

  uint64_t ml, mc, mr;
  ml = Particles[index - 1].Encoded;
  mc = Particles[index].Encoded;
  mr = Particles[index + 1].Encoded;

  if (ml == mc && mc == mr) {
    for (; (index < N) && (Particles[index].Encoded != Particles[index + 1].Encoded); index++) {}
    return make_uint2(iind, index);
  } else {
    uint2 lr = make_uint2(__clz(mc ^ ml), __clz(mc ^ mr));
    if (lr.x > lr.y) {
      dir = -1;
      d_min = lr.y;
    } else {
      dir = 1;
      d_min = lr.x;
    }
  }
  uint32_t l_max = 2;
  int64_t testindex = index + l_max*dir;
  while (testindex < N && testindex >= 0 ? __clz(mc ^ Particles[testindex].Encoded)>d_min : false) {
    l_max *= 2;
    testindex = index + l_max*dir;
  }

  uint32_t l, t, splitPrefix;
  int64_t nt;
  for (uint32_t div = 2; l_max / div >= 1; div *= 2) {
    t = l_max / div;
    nt = index + (l + t)*dir;
    if (nt < N) {
      splitPrefix = __clz(mc ^ Particles[nt].Encoded);
      if (splitPrefix > d_min) { l += t; }
    }
  }

  return dir > 0 ? make_uint2(index, index + l*dir) : make_uint2(index + l*dir, index);
}

__device__ uint32_t FindSplit(uint32_t N, KerParticle* Particles, uint32_t lp, uint32_t rp)
{
  uint64_t lc, rc;
  lc = Particles[lp].Encoded; rc = Particles[rp].Encoded;

  if (lc == rc) return lp;

  uint64_t prefix = __clz(lc ^ rc);
  uint32_t split = lp;
  int64_t step = rp - lp;
  int64_t ns;
  uint64_t sc, sp;
  while (step > 1) {
    step = (step + 1) >> 1;
    ns = split + step;

    if (ns < rp) {
      sc = Particles[ns].Encoded;
      sp = __clz(lc ^ sc);
      if (sp > prefix) { split = ns; }
    }
  }

  return split;
}

__device__ void GenerateNode(uint32_t N, KerParticle* Particles, KerNode* Tree, KerNode* Leafs, uint32_t index)
{
  uint2 r = CalculateRange(N, Particles, index);
  uint32_t split = FindSplit(N, Particles, r.x, r.y);

  KerNode *n1, *n2;
  n1 = (split == r.x ? Leafs : Tree) + split;
  n2 = (split + 1 == r.y ? Leafs : Tree) + split + 1;

  Tree[index].LeftNode = n1;
  Tree[index].RightNode = n2;

  n1->ParentNode = Tree + index;
  n2->ParentNode = Tree + index;
}

__global__ void GenerateOctree(uint32_t N, KerParticle * Particles, KerNode * Tree, KerNode * Leaves)
{
  uint32_t i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < N) { GenerateNode(N, Particles, Tree, Leaves, i); }
}

__global__ void UpdateData(uint32_t N, KerParticle * Particles, float4 * Data)
{
  uint32_t i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < N) { Data[i] = Particles[i].Position; }
}


NBodyKernel::~NBodyKernel()
{
  if (Initialized) {
    cudaFree(d_RawData);
    cudaFree(d_KerParticles);
    cudaFree(d_KerTree);
    cudaFree(d_KerLeaves);
    delete[] RawData;
    delete[] KerParticles;
  }
}

void NBodyKernel::Initialize(uint32_t PartSize, float4* PartArray)
{
  if (Initialized) {
    cudaFree(d_RawData);
    cudaFree(d_KerParticles);
    cudaFree(d_KerTree);
    cudaFree(d_KerLeaves);
    delete[] RawData;
    delete[] KerParticles;
  }

  NParticles = PartSize;
  KerParticles = new KerParticle[NParticles];
  RawData = new float4[NParticles];

  for (uint32_t i = 0; i < NParticles; i++) { RawData[i] = PartArray[i]; }

  cudaMalloc(&d_RawData, sizeof(float4)*NParticles);
  cudaMalloc(&d_KerParticles, sizeof(KerParticle)*NParticles);
  cudaMalloc(&d_KerLeaves, sizeof(KerNode)*NParticles);
  cudaMalloc(&d_KerTree, sizeof(KerNode)*(NParticles - 1));
  
  cudaMemcpy(d_RawData, RawData, sizeof(float4)*NParticles, cudaMemcpyHostToDevice);

  ParticleAlloc<<<(NParticles + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(NParticles, d_RawData, d_KerParticles, d_KerLeaves);

  Initialized = true;
}

void NBodyKernel::CopyEncodedToHost()
{
  if (!Initialized) return;
  UpdateData<<<(NParticles + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> >(NParticles, d_KerParticles, d_RawData);
  cudaMemcpy(RawData, d_RawData, sizeof(float4)*NParticles, cudaMemcpyDeviceToHost);

  for (uint32_t i = 0; i < NParticles; i++) { KerParticles[i].Position = RawData[i]; }
}

void NBodyKernel::GPUBuildOctree()
{
  if (!Initialized) return;
  thrust::sort(d_KerParticles, d_KerParticles + NParticles,
    [](const KerParticle &a, const KerParticle &b) -> bool { return a.Encoded < b.Encoded; });
}
