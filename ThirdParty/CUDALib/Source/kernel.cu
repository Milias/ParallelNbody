#include "kernel.cuh"
#include "kernel_wrapper.h"

#ifndef __CUDACC__
  #define __CUDACC__
#endif

// CUDA
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_functions.h>

// Thrust
#include <thrust/system_error.h>
#include <thrust/execution_policy.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/unique.h>
#include <thrust/count.h>
#include <thrust/extrema.h>

// Constants
#define THREADS_PER_BLOCK 256
#define G_CONSTANT 0.0
#define EPSILON 0.1

// Debug
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
  if (code != cudaSuccess)
  {
    OutputDebugString(("GPUassert: " + std::string(cudaGetErrorString(code)) + " " + file + " " + std::to_string(line) + "\n").c_str());
    if (abort) exit(code);
  }
}

// UTIL

__host__ __device__ uint64_t ReverseChar(uint64_t x);
__host__ __device__ uint64_t Reverse(uint64_t x);
__host__ __device__ uint64_t SplitBy3(uint64_t x);
__host__ __device__ uint64_t MagicBits(uint64_t x, uint64_t y, uint64_t z);
__host__ __device__ uint64_t KerEncode(float4 v);

// Host/Device Functions

__host__ __device__ void MortonEncoding(uint32_t i, KerParticle * Particles, uint64_t * Encoded, uint64_t * Counts, float3 Size, float MaxSize);
__host__ __device__ void ComputeNodeKeys(uint32_t i, uint64_t * Counts, uint64_t * Encoded, uint32_t k);
__host__ __device__ void InitializeOctree(uint32_t N, uint32_t i, TreeCell * OctreeCell, KerParticle * Particles, uint64_t * Encoded, uint32_t * Temp, uint64_t * values, uint32_t CellSize, uint32_t TotalSize, uint32_t k);
__host__ __device__ void LinkOctree(uint32_t N, uint32_t i, TreeCell * Octree, uint32_t * d_ChildrenCounter, uint64_t * Range, uint32_t dLevel, uint32_t TotalSize);
__host__ __device__ float4 CenterOfMass(uint32_t i, TreeCell * Octree, KerParticle * Particles);
__host__ __device__ float Distance(float4 &v1, float4 &v2);
__host__ __device__ float3 Acceleration(float4 &v1, float4 &v2, float d);
__host__ __device__ void AddAcceleration(KerParticle * p, float4 &v, float d);
__host__ __device__ void ForceOnParticle(uint32_t pIndex, KerParticle * Particles, TreeCell * Node, float Size);
__host__ __device__ void IntegrateParticle(uint32_t Index, KerParticle * Particles, float dt);

// Kernels

__global__ void KerBoundary(float3 * Size, KerParticle * ax, KerParticle * ay, KerParticle * az, KerParticle * bx, KerParticle * by, KerParticle * bz);
__global__ void KerAllocateData(uint32_t N, float4 * DataPos, float3 * DataVel, KerParticle * Particles, uint64_t * Encoded, uint64_t * Counts, float3 Size, float MaxSize);
__global__ void KerEncodeParticles(uint32_t N, KerParticle * Particles, uint64_t * Encoded, uint64_t * Counts, float3 * Size, float MaxSize);
__global__ void KerComputeNodes(uint32_t N, uint64_t * Counts, uint64_t * Encoded, uint32_t k);
__global__ void KerUniques(uint32_t nNodes, uint32_t nParticles, uint64_t * Counts, uint64_t * Freq, uint64_t * Values);
__global__ void KerInitializeOctree(uint32_t N, TreeCell * Octree, KerParticle * Particles, uint64_t * Encoded, uint32_t * Temp, uint64_t * values, uint32_t CellSize, uint32_t TotalSize, uint32_t k);
__global__ void KerCopyLinearOctree(uint32_t N, uint32_t dLevel, uint64_t * Offset, TreeCell ** Octree, TreeCell * LinearOctree);
__global__ void KerLinkingOctree(uint32_t N, TreeCell * Octree, uint32_t * d_ChildrenCounter, uint64_t * Range, uint32_t dLevel, uint32_t TotalSize);
__global__ void KerCenterOfMass(uint32_t N, TreeCell * Octree, KerParticle * Particles);
__global__ void KerForces(uint32_t pIndex, uint32_t N, KerParticle * Particles, TreeCell * Octree, float3 * Size);
__global__ void KerIntegration(uint32_t N, KerParticle * Particles, float dt);
__global__ void KerUpdateData(uint32_t N, KerParticle * Particles, float4 * Data);

/***** CUDA host/device Functions *****/

__host__ __device__ uint64_t ReverseChar(uint64_t x)
{
  return (x * 0x0202020202ULL & 0x010884422010ULL) % 1023;
}

__host__ __device__ uint64_t Reverse(uint64_t x)
{
  uint8_t inByte0 = (x & 0xFF);
  uint8_t inByte1 = (x & 0xFF00) >> 8;
  uint8_t inByte2 = (x & 0xFF0000) >> 16;
  uint8_t inByte3 = (x & 0xFF000000) >> 24;
  uint8_t inByte4 = (x & 0xFF00000000) >> 32;
  uint8_t inByte5 = (x & 0xFF0000000000) >> 40;
  uint8_t inByte6 = (x & 0xFF000000000000) >> 48;
  uint8_t inByte7 = (x & 0xFF00000000000000) >> 56;
  return (ReverseChar(inByte0) << 56) | (ReverseChar(inByte1) << 48) | (ReverseChar(inByte2) << 40) | (ReverseChar(inByte3) << 32) | (ReverseChar(inByte4) << 24) | (ReverseChar(inByte5) << 16) | (ReverseChar(inByte6) << 8) | (ReverseChar(inByte7));
}

__host__ __device__ uint64_t SplitBy3(uint64_t x)
{
  x &= 0x1fffff;
  x = (x | (x << 32)) & 0x1f00000000ffff;
  x = (x | (x << 16)) & 0x1f0000ff0000ff;
  x = (x | (x << 8)) & 0x100f00f00f00f00f;
  x = (x | (x << 4)) & 0x10c30c30c30c30c3;
  x = (x | (x << 2)) & 0x1249249249249249;
  return x;
}

__host__ __device__ uint64_t MagicBits(uint64_t x, uint64_t y, uint64_t z)
{
  return Reverse(SplitBy3(x) | SplitBy3(y) << 1 | SplitBy3(z) << 2);
}

__host__ __device__ uint64_t KerEncode(float4 v)
{
  return MagicBits((uint64_t)(v.x), (uint64_t)(v.y), (uint64_t)(v.z));
}

__host__ __device__ void MortonEncoding(uint32_t i, KerParticle * Particles, uint64_t * Encoded, uint64_t * Counts, float3 Size, float MaxSize)
{
  float4 t = make_float4(0, 0, 0, 0);
  t.x = Particles[i].Position.x / Size.x * MaxSize;
  t.y = Particles[i].Position.y / Size.y * MaxSize;
  t.z = Particles[i].Position.z / Size.z * MaxSize;
  Encoded[i] = KerEncode(t);
  Counts[i] = 0;
}

__host__ __device__ void ComputeNodeKeys(uint32_t i, uint64_t * Counts, uint64_t * Encoded, uint32_t k)
{
  Counts[i] = Encoded[i] & Reverse((1i64 << 3 * k) - 1);
}

__host__ __device__ void InitializeOctree(uint32_t N, uint32_t i, TreeCell * OctreeCell, KerParticle * Particles, uint64_t * Encoded, uint32_t * Temp, uint64_t * values, uint32_t CellSize, uint32_t TotalSize, uint32_t k)
{
  OctreeCell[i].Range = make_uint2(Temp[i], i == CellSize - 1 ? N : Temp[i + 1]);
  OctreeCell[i].Morton = Reverse(values[i]);
  OctreeCell[i].Level = k;

  if (OctreeCell[i].Range.x + 1 == OctreeCell[i].Range.y) {
    OctreeCell[i].Leaf = true;
    OctreeCell[i].Particle = Particles + Temp[i];
    OctreeCell[i].Particle->Parent = TotalSize + i;
  }
  //CenterOfMass(i, OctreeCell, Particles);
}

__host__ __device__ void LinkOctree(uint32_t N, uint32_t i, TreeCell * Octree, uint32_t * ChildrenCounter, uint64_t * Range, uint32_t dLevel, uint32_t TotalSize)
{
  register uint64_t x0, x1, x2, x3;
  register TreeCell node;
  if (i == 0) {
    Octree[0].Range = make_uint2(0, N);
  } else {
    node = Octree[i];
    x0 = Range[node.Level - 1];
    x1 = Range[node.Level];
    x2 = Range[node.Level + 1];
    x3 = (node.Level + 2 >= dLevel ? TotalSize : Range[node.Level + 2]);
    for (uint64_t j = x0; j < x1; j++) {
      if (Octree[j].Leaf) continue;      
      if ((node.Morton << 3) == Octree[j].Morton) {
#if defined( __CUDA_ARCH__ )
        //atomicAdd(ChildrenCounter + i, 1);
#else
        ChildrenCounter[i]++;
#endif
        node.Parent = j;
        break;
      }
    }
    /*
    if (!Octree[i].Leaf) {
      for (uint32_t j = Range[Octree[i].Level + 1]; j < ; j++) {
        if ((Octree[j].Morton << 3) == Octree[i].Morton) {
          Octree[i].Child = j;
          break;
        }
      }
    }
    */
  }
}

__host__ __device__ float4 CenterOfMass(uint32_t i, TreeCell * Octree, KerParticle * Particles)
{
  register uint64_t x0, x1;
  register TreeCell node = Octree[i];
  register KerParticle p;
  x0 = node.Range.x; x1 = node.Range.y;
  node.Position = make_float4(0, 0, 0, 0);
  for (uint32_t j = x0; j < x1; j++) {
    p = Particles[j];
    node.Position.x += p.Position.x * p.Position.w;
    node.Position.y += p.Position.y * p.Position.w;
    node.Position.z += p.Position.z * p.Position.w;
    node.Position.w += p.Position.w;
  }
  node.Position.x /= node.Position.w;
  node.Position.y /= node.Position.w;
  node.Position.z /= node.Position.w;
  return node.Position;
}

__host__ __device__ float Distance(float4 &v1, float4 &v2)
{
  return sqrtf(pow(v1.x - v2.x, 2) + pow(v1.y - v2.y, 2) + pow(v1.z - v2.z, 2));
}

// acceleration of v1 produced by v2, v1 -> v2.
__host__ __device__ float3 Acceleration(float4 &v1, float4 &v2, float d)
{
  float t = G_CONSTANT * v2.w / pow(d + EPSILON, 3);
  return make_float3(t*(v2.x - v1.x), t*(v2.y - v1.y), t*(v2.z - v1.z));
}

__host__ __device__ void AddAcceleration(KerParticle * p, float4 &v, float d)
{
  float3 a = Acceleration(p->Position, v, d);
  p->Acceleration.x += a.x;
  p->Acceleration.y += a.y;
  p->Acceleration.z += a.z;
}

__host__ __device__ void ForceOnParticle(uint32_t pIndex, KerParticle * Particles, TreeCell * Node, float Size)
{
  register float d = Distance(Particles[pIndex].Position, Node->Position);
  if (Node->Leaf) {
    AddAcceleration(Particles + pIndex, Node->Position, d);
  } else if (Size < d * (1 << Node->Level)) {
    AddAcceleration(Particles + pIndex, Node->Position, d);
  }
}

__host__ __device__ void IntegrateParticle(uint32_t Index, KerParticle * Particles, float dt)
{
  Particles[Index].Velocity.x += Particles[Index].Acceleration.x * dt;
  Particles[Index].Velocity.y += Particles[Index].Acceleration.y * dt;
  Particles[Index].Velocity.z += Particles[Index].Acceleration.z * dt;
  Particles[Index].Position.x += Particles[Index].Velocity.x * dt;
  Particles[Index].Position.y += Particles[Index].Velocity.y * dt;
  Particles[Index].Position.z += Particles[Index].Velocity.z * dt;
}

/***** CUDA Kernels *****/

__global__ void KerBoundary(uint32_t N, KerParticle * Particles, float3 * Size)
{
  register float3 a, b;
  register uint32_t i = threadIdx.x;
  if (i == 0) {
    a.x = thrust::max_element(thrust::device, Particles, Particles + N, CompareParticleX())->Position.x;
    b.x = thrust::min_element(thrust::device, Particles, Particles + N, CompareParticleX())->Position.x;
    Size->x = a.x - b.x;
  } else if (i == 1) {
    a.y = thrust::max_element(thrust::device, Particles, Particles + N, CompareParticleY())->Position.y;
    b.y = thrust::min_element(thrust::device, Particles, Particles + N, CompareParticleY())->Position.y;
    Size->y = a.y - b.y;
  } else {
    a.z = thrust::max_element(thrust::device, Particles, Particles + N, CompareParticleZ())->Position.z;
    b.z = thrust::min_element(thrust::device, Particles, Particles + N, CompareParticleZ())->Position.z;
    Size->z = a.z - b.z;
  }
}

__global__ void KerAllocateData(uint32_t N, float4 * DataPos, float3 * DataVel, KerParticle * Particles, uint64_t * Encoded, uint64_t * Counts, float3 * Size, float MaxSize)
{
  uint32_t i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < N) {
    Particles[i].Position.x = DataPos[i].x;
    Particles[i].Position.y = DataPos[i].y;
    Particles[i].Position.z = DataPos[i].z;
    Particles[i].Position.w = DataPos[i].w;
    Particles[i].Velocity.x = DataVel[i].x;
    Particles[i].Velocity.y = DataVel[i].y;
    Particles[i].Velocity.z = DataVel[i].z;
    MortonEncoding(i, Particles, Encoded, Counts, *Size, MaxSize);
  }
}

__global__ void KerEncodeParticles(uint32_t N, KerParticle * Particles, uint64_t * Encoded, uint64_t * Counts, float3 * Size, float MaxSize)
{
  uint32_t i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < N) { MortonEncoding(i, Particles, Encoded, Counts, *Size, MaxSize); }
}

__global__ void KerComputeNodes(uint32_t N, uint64_t * Counts, uint64_t * Encoded, uint32_t k)
{
  uint32_t i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < N) { ComputeNodeKeys(i, Counts, Encoded, k); }
}

__global__ void KerUniques(uint32_t nNodes, uint32_t nParticles, uint64_t * Counts, uint64_t * Freq, uint64_t * Values)
{
  uint32_t i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < nNodes) Freq[i] = thrust::count(thrust::device, Counts, Counts + nParticles, Values[i]);
}

__global__ void KerInitializeOctree(uint32_t N, TreeCell * Octree, KerParticle * Particles, uint64_t * Encoded, uint32_t * Temp, uint64_t * values, uint32_t CellSize, uint32_t TotalSize, uint32_t k)
{
  uint32_t i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < N) { 
    InitializeOctree(N, i, Octree, Particles, Encoded, Temp, values, CellSize, TotalSize, k);
  }
}

__global__ void KerCopyLinearOctree(uint32_t N, uint32_t dLevel, uint64_t * Offset, TreeCell ** Octree, TreeCell * LinearOctree)
{
  register uint32_t i = blockIdx.x*blockDim.x + threadIdx.x;
  register uint32_t osInd;
  if (i < N) {
    for (uint32_t j = 0; j < dLevel; j++) {
      if (Offset[j] > i) {
        osInd = j - 1;
        break;
      }
    }
    if (osInd > 0) LinearOctree[i] = Octree[osInd][Offset[osInd] + i];
  }
}

__global__ void KerLinkingOctree(uint32_t N, TreeCell * Octree, uint32_t * ChildrenCounter, uint64_t * Offset, uint32_t dLevel, uint32_t TotalSize)
{
  uint32_t i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < N) { LinkOctree(N, i, Octree, ChildrenCounter, Offset, dLevel, TotalSize); }
}

__global__ void KerCenterOfMass(uint32_t N, TreeCell * Octree, KerParticle * Particles)
{
  uint32_t i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < N) { Octree[i].Position = CenterOfMass(i, Octree, Particles); }
}

__global__ void KerForces(uint32_t pIndex, uint32_t N, KerParticle * Particles, TreeCell * Octree, float3 * Size)
{
  uint32_t i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < N) { ForceOnParticle(pIndex, Particles, Octree + i, max(Size->x, max(Size->y, Size->z))); }
}

__global__ void KerIntegration(uint32_t N, KerParticle * Particles, float dt)
{
  uint32_t i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < N) { IntegrateParticle(i, Particles, dt); }
}

__global__ void KerUpdateData(uint32_t N, KerParticle * Particles, float4 * Data)
{
  uint32_t i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < N) { Data[i] = Particles[i].Position; }
}

NBodyKernel::~NBodyKernel()
{
  CleanCPU();
  CleanGPU();
}

void NBodyKernel::CleanCPU()
{
  if (InitializedCPU) {
    delete[] OctreeCells;
    delete[] TreeCellsSizes;
    delete[] KerParticles;
    delete[] Counts;
    delete[] Encoded;
    delete[] Temp;
    delete[] values;
    delete[] temp64;
  }
}

void NBodyKernel::CleanGPU()
{
  if (InitializedGPU) {
    delete[] TreeCellsSizes;
    delete[] OctreeCells;
    cudaFree(d_temp64);
    cudaFree(d_Temp);
    cudaFree(d_RawDataPos);
    cudaFree(d_RawDataVel);
    cudaFree(d_KerParticles);
    cudaFree(d_Encoded);
    cudaFree(d_Counts);
    cudaFree(d_values);
    cudaFree(d_Size);
  }
}

void NBodyKernel::InitializeCPU(uint32_t PartSize, float4* p, float3* v, float dt)
{
  if (InitializedGPU) return;
  if (!InitializedCPU) {
    OctreeCells = new TreeCell*[MaxDepth];
    TreeCellsSizes = new uint32_t[MaxDepth];
    TreeCellsSizes[0] = 1;
  }

  if (InitializedCPU && (PartSize != NParticles)) {
    delete[] KerParticles;
    delete[] Counts;
    delete[] Encoded;
    delete[] Temp;
    delete[] values;
    delete[] temp64;
  }

  if (!InitializedCPU || (PartSize != NParticles)) {
    NParticles = PartSize;
    KerParticles = new KerParticle[NParticles];
    Counts = new uint64_t[NParticles];
    Encoded = new uint64_t[NParticles];
    Temp = new uint32_t[NParticles];
    values = new uint64_t[NParticles];
    temp64 = new uint64_t[NParticles];
  }

  CubeCornerA = make_float3(0, 0, 0);
  CubeCornerB = make_float3(0, 0, 0);
  deltaTime = dt;
  for (uint32_t i = 0; i < NParticles; i++) {
    KerParticles[i].Position = p[i];
    KerParticles[i].Velocity = v[i];
    KerParticles[i].Acceleration = make_float3(0, 0, 0);
  }

  InitializedCPU = true;
}

void NBodyKernel::InitializeGPU(uint32_t PartSize, float4* p, float3* v, float dt)
{
  if (InitializedCPU) return;
  if (!InitializedGPU) {
    cudaMalloc(&d_Size, sizeof(float3));
    TreeCellsSizes = new uint32_t[MaxDepth];
    OctreeCells = new TreeCell*[MaxDepth];
  }

  if (InitializedGPU && (PartSize != NParticles)) {
    delete[] RawDataPos;
    cudaFree(d_RawDataPos);
    cudaFree(d_RawDataVel);
    cudaFree(d_KerParticles);
    cudaFree(d_Encoded);
    cudaFree(d_Counts);
    cudaFree(d_values);
    cudaFree(d_temp64);
    cudaFree(d_Temp);
  }

  if (!InitializedGPU || (PartSize != NParticles)) {
    NParticles = PartSize;
    RawDataPos = new float4[NParticles];

    cudaMalloc(&d_RawDataPos, sizeof(float4)*NParticles);
    cudaMalloc(&d_RawDataVel, sizeof(float3)*NParticles);
    cudaMalloc(&d_KerParticles, sizeof(KerParticle)*NParticles);
    cudaMalloc(&d_Encoded, sizeof(uint64_t)*NParticles);
    cudaMalloc(&d_Counts, sizeof(uint64_t)*NParticles);
    cudaMalloc(&d_values, sizeof(uint64_t)*NParticles);
    cudaMalloc(&d_temp64, sizeof(uint64_t)*NParticles);
    cudaMalloc(&d_Temp, sizeof(uint32_t)*NParticles);
  }

  CubeCornerA = make_float3(0, 0, 0);
  CubeCornerB = make_float3(0, 0, 0);
  deltaTime = dt;

  cudaMemcpy(d_RawDataPos, p, sizeof(float4)*NParticles, cudaMemcpyHostToDevice);
  cudaMemcpy(d_RawDataVel, v, sizeof(float3)*NParticles, cudaMemcpyHostToDevice);

  KerAllocateData<<<(NParticles + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(NParticles, d_RawDataPos, d_RawDataVel, d_KerParticles, d_Encoded, d_Counts, d_Size, MaxSize);
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());
  InitializedGPU = true;
}

void NBodyKernel::CopyEncodedToHost()
{
  if (!InitializedGPU) return;
  KerUpdateData<<<(NParticles + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(NParticles, d_KerParticles, d_RawDataPos);
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaMemcpy(RawDataPos, d_RawDataPos, sizeof(float4)*NParticles, cudaMemcpyDeviceToHost));
}

void NBodyKernel::GPUBuildOctree()
{
  if (!InitializedGPU) return;
  thrust::device_ptr<KerParticle> dp_Particles(d_KerParticles);
  thrust::device_ptr<uint64_t> dp_Encoded(d_Encoded);
  thrust::device_ptr<uint64_t> dp_Counts(d_Counts);
  thrust::device_ptr<uint64_t> dp_values(d_values);
  thrust::device_ptr<uint64_t> dp_temp64(d_temp64);
  thrust::device_ptr<uint32_t> dp_Temp(d_Temp);
  
  /////////////////////////////
  // BOUNDING BOX
  /////////////////////////////

  KerBoundary<<<1, 3>>>(NParticles, d_KerParticles, d_Size);
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());

  /////////////////////////////
  // MORTON ENCODING AND SORT
  /////////////////////////////

  KerEncodeParticles<<<(NParticles + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(NParticles, d_KerParticles, d_Encoded, d_Counts, d_Size, MaxSize);
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());
  thrust::sort_by_key(dp_Encoded, dp_Encoded + NParticles, dp_Particles);

  thrust::fill(TreeCellsSizes + 1, TreeCellsSizes + MaxDepth, 0);
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());
  /////////////////////////////
  // OCTREE BUILDING
  /////////////////////////////

  uint32_t CellSize = 0, dLevel = 1;
  totalSize = 1;
  for (uint32_t k = 1; k < MaxDepth && CellSize < NParticles; k++) {
    
    KerComputeNodes<<<(NParticles + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(NParticles, d_Counts, d_Encoded, k);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    dLevel++;
    CellSize = thrust::unique_copy(dp_Counts, dp_Counts + NParticles, dp_values) - dp_values;
    TreeCellsSizes[k] = CellSize;
    if (CellSize < NParticles) {
      KerUniques<<<(CellSize + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(CellSize, NParticles, d_Counts, d_temp64, d_values);
    } else {
      thrust::fill(dp_temp64, dp_temp64 + NParticles, 1);
    }
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    thrust::exclusive_scan(dp_temp64, dp_temp64 + CellSize, dp_Temp);
    //OutputDebugString(("Total: " + std::to_string(k) + "\n").c_str());

    cudaMalloc(OctreeCells + k, sizeof(TreeCell)*CellSize);

    KerInitializeOctree<<<(CellSize + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(CellSize, OctreeCells[k], d_KerParticles, d_Encoded, d_Temp, d_values, CellSize, totalSize, k);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    totalSize += CellSize;
  }
  /////////////////////////////
  // OCTREE LINKING
  /////////////////////////////

  cudaMalloc(&d_FinalOctreeCells, sizeof(TreeCell)*totalSize);
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());
  uint64_t *Offset = new uint64_t[dLevel];
  thrust::exclusive_scan(TreeCellsSizes, TreeCellsSizes + dLevel, Offset);

  uint64_t *d_Offset;
  cudaMalloc(&d_Offset, sizeof(uint64_t)*dLevel);
  cudaMemcpy(d_Offset, Offset, sizeof(uint64_t)*dLevel, cudaMemcpyHostToDevice);
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());

  cudaMalloc(&d_OctreeCells, sizeof(TreeCell*)*dLevel);
  cudaMemcpy(d_OctreeCells, OctreeCells, sizeof(TreeCell*)*dLevel, cudaMemcpyHostToDevice);
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());

  cudaMalloc(&d_ChildrenCounter, sizeof(uint32_t)*totalSize);
  thrust::device_ptr<uint32_t> dp_ChildrenCounter(d_ChildrenCounter);
  thrust::fill(dp_ChildrenCounter, dp_ChildrenCounter + totalSize, 0);
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());

  KerCopyLinearOctree<<<(totalSize + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(totalSize, dLevel, d_Offset, d_OctreeCells, d_FinalOctreeCells);
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());
  cudaFree(d_OctreeCells);
  for (uint32_t i = 1; i < dLevel; i++) { cudaFree(OctreeCells[i]); }

  KerLinkingOctree<<<(totalSize + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(NParticles,  d_FinalOctreeCells, d_ChildrenCounter, d_Offset, dLevel, totalSize);
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());
  cudaFree(d_Offset);
  
  /////////////////////////////
  // CoM CALCULATION
  /////////////////////////////

  KerCenterOfMass<<<(totalSize + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(totalSize, d_FinalOctreeCells, d_KerParticles);
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());


  /////////////////////////////
  // FORCE CALCULATION
  /////////////////////////////

  for (uint32_t i = 0; i < NParticles; i++) {
    KerForces<<<(totalSize + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(i, totalSize, d_KerParticles, d_FinalOctreeCells, d_Size);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
  }

  /////////////////////////////
  // INTEGRATE NEW POSITIONS
  /////////////////////////////

  KerIntegration<<<(NParticles + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(NParticles, d_KerParticles, deltaTime);
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());
}

void NBodyKernel::CPUBuildOctree()
{
  if (!InitializedCPU) return;
  /////////////////////////////
  // BOUNDING BOX
  /////////////////////////////
  
  CubeCornerA.x = thrust::max_element(KerParticles, KerParticles + NParticles, [](const KerParticle &a, const KerParticle &b) -> bool { return a.Position.x < b.Position.x; })->Position.x;
  CubeCornerA.y = thrust::max_element(KerParticles, KerParticles + NParticles, [](const KerParticle &a, const KerParticle &b) -> bool { return a.Position.y < b.Position.y; })->Position.y;
  CubeCornerA.z = thrust::max_element(KerParticles, KerParticles + NParticles, [](const KerParticle &a, const KerParticle &b) -> bool { return a.Position.z < b.Position.z; })->Position.z;
  CubeCornerB.x = thrust::min_element(KerParticles, KerParticles + NParticles, [](const KerParticle &a, const KerParticle &b) -> bool { return a.Position.x < b.Position.x; })->Position.x;
  CubeCornerB.y = thrust::min_element(KerParticles, KerParticles + NParticles, [](const KerParticle &a, const KerParticle &b) -> bool { return a.Position.y < b.Position.y; })->Position.y;
  CubeCornerB.z = thrust::min_element(KerParticles, KerParticles + NParticles, [](const KerParticle &a, const KerParticle &b) -> bool { return a.Position.z < b.Position.z; })->Position.z;

  Size = make_float3(CubeCornerA.x - CubeCornerB.x, CubeCornerA.y - CubeCornerB.y, CubeCornerA.z - CubeCornerB.z);
  //OutputDebugString(("\n\nSize: " + std::to_string(Size.x) + ", " + std::to_string(Size.y) + ", " + std::to_string(Size.z) + ", " + "\n").c_str());

  /////////////////////////////
  // MORTON ENCODING AND SORT
  /////////////////////////////

  for (uint32_t i = 0; i < NParticles; i++) {
    MortonEncoding(i, KerParticles, Encoded, Counts, Size, MaxSize);
  }

  thrust::sort_by_key(Encoded, Encoded + NParticles, KerParticles);
  //assert(thrust::unique(Encoded, Encoded + NParticles) - Encoded == NParticles);
  /*
  char msg[65]; msg[64] = '\0';
  for (uint32_t i = 0; i < NParticles; i++) {
    for (int j = 0; j < 64; j++) { msg[j] = (Encoded[i] & 1i64 << j) ? '1' : '0'; }
    OutputDebugString(("Index: " + std::to_string(i) + ", \t" + msg + "\n").c_str());
  }
  */
  thrust::fill(TreeCellsSizes, TreeCellsSizes + 1, 1);
  thrust::fill(TreeCellsSizes + 1, TreeCellsSizes + MaxDepth, 0);

  /////////////////////////////
  // OCTREE BUILDING
  /////////////////////////////

  uint32_t t2 = 0, c = 1;
  totalSize = 1;
  for (uint32_t k = 1; k < MaxDepth && t2 < NParticles; k++) {

    for (uint32_t i = 0; i < NParticles; i++) { ComputeNodeKeys(i, Counts, Encoded, k); }

    c++;
    t2 = thrust::unique_copy(Counts, Counts + NParticles, values) - values;
    TreeCellsSizes[k] = t2;
    if (t2 < NParticles) {
      for (uint32_t i = 0; i < t2; i++) { temp64[i] = thrust::count(Counts, Counts + NParticles, values[i]); }
    } else { thrust::fill(temp64, temp64 + NParticles, 1); }
    thrust::exclusive_scan(temp64, temp64 + t2, Temp);

    OctreeCells[k] = new TreeCell[t2];

    //OutputDebugString(("Total: " + std::to_string(t2) + "\n").c_str());
    for (uint32_t i = 0; i < t2; i++) {
      InitializeOctree(NParticles, i, OctreeCells[k], KerParticles, Encoded, Temp, values, t2, totalSize, k);
    }
    totalSize += t2;
  }

  /////////////////////////////
  // OCTREE LINKING
  /////////////////////////////
  /*
  FinalOctreeCells = new TreeCell[totalSize];
  ChildrenCounter = new uint32_t[totalSize];
  uint64_t *temp2 = new uint64_t[c];
  thrust::exclusive_scan(TreeCellsSizes, TreeCellsSizes + c, temp2);
  //OutputDebugString(("Octree nodes: " + std::to_string(totalSize) + "\n").c_str());
  
  for (uint32_t i = 0; i < c; i++) {
    thrust::copy(OctreeCells[i], OctreeCells[i] + TreeCellsSizes[i], FinalOctreeCells + temp2[i]);
    //delete[] OctreeCells[i];
  }

  for (uint32_t i = 0; i < totalSize; i++) {
    LinkOctree(NParticles, i, FinalOctreeCells, ChildrenCounter, temp2, c, totalSize);
  }
  */
  /*
  for (uint32_t i = 0; i < totalSize; i++) {
    OutputDebugString(("Index: " + std::to_string(i) + " childs: " + std::to_string(FinalOctreeCells[i].ChildCounter) + "\n").c_str());
    for (uint32_t k = 0; k < FinalOctreeCells[i].ChildCounter; k++) {
      for (int j = 0; j < 64; j++) { msg[j] = (FinalOctreeCells[FinalOctreeCells[i].Child + k].Morton & 1i64 << j) ? '1' : '0'; }
      OutputDebugString(("\t\t" + std::string(msg) + "\n").c_str());
    }
  }
  */
  /////////////////////////////
  // CoM CALCULATION
  /////////////////////////////

  //for (uint32_t i = 0; i < totalSize; i++) {
  //  CenterOfMass(i, FinalOctreeCells, KerParticles);
  //}

  /////////////////////////////
  // FORCE CALCULATION
  /////////////////////////////
  
  for (uint32_t i = 0; i < NParticles; i++) {
    for (uint32_t j = 0; j < totalSize; j++) {
      ForceOnParticle(i, KerParticles, FinalOctreeCells + j, max(Size.x, max(Size.y, Size.z)));
    }
  }

  /////////////////////////////
  // INTEGRATE NEW POSITIONS
  /////////////////////////////

  for (uint32_t i = 0; i < NParticles; i++) {
    IntegrateParticle(i, KerParticles, deltaTime);
  }

}
