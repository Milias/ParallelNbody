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
#define N_BLOCK 64
#define G_CONSTANT .0
#define EPSILON 0.1
#define BENCH_FRAMES 500
#define BENCH_AMOUNT 7

// Debug
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

// UTIL
/*
  Reverses a 8bit integer, used in conjunction with Reverse(uint64_t x).

  Input: 64 bit integer.
  Output: first 8 bits of input will be reversed.
*/
__host__ __device__ uint64_t ReverseChar(uint64_t x);

/*
  Reverses a 64 bit integer.
  Input: 64 bit integer.
  Output: reversed input.
*/
__host__ __device__ uint64_t Reverse(uint64_t x);

/*
  Takes a 64 bit integer and splits it leaving 2 bits
  between each one.

  Example: 10101 -> 001 000 001 000 001
*/
__host__ __device__ uint64_t SplitBy3(uint64_t x);

/*
  Takes three 64 bit integers and by using SplitBy3 it
  returns the interleaved version.

  Example: 0010 0100 1000 -> 000 000 001 000
*/
__host__ __device__ uint64_t MagicBits(uint64_t x, uint64_t y, uint64_t z);

/*
  Convenience function. Takes a float4 vector and returns
  the corresponding Morton key.
*/
__host__ __device__ uint64_t KerEncode(float4 v);

// Host/Device Functions

/*
  Converts Particles[i] position to Morton key, taking into account
  the size of the box.

  Inputs:
    Index of the particle.
    Particles array.
    Encoded keys array.
    Size of the Box.
    MinX, MinY, MinZ: particles with minimum value in each coordinate.
    Maximum size of the box, 2^20.
*/
__host__ __device__ void MortonEncoding(uint32_t i, KerParticle * Particles, uint64_t * Encoded, float3 * Size, KerParticle * MinX, KerParticle * MinY, KerParticle * MinZ, float MaxSize);

/*
  Computes masked keys at building step.

  Inputs:
    Index of the particle.
    Masked keys array.
    Encoded Morton keys array.
    Level.
*/
__host__ __device__ void ComputeNodeKeys(uint32_t i, MaskedKey * Counts, uint64_t * Encoded, uint32_t k);

/*
  Takes a Linear Octree and, for each node, looks for
  its parent and its first child. When it's finished
  every node is linked.

  Input:
    Total number of particles.
    Index of the node.
    Pointer to the root of the octree.
    Array where each node number of children is contained.
    Array where the ranges of each level in the octree are contained.
    Depth of the tree.
    Number of nodes in the linear octree.
*/
__host__ __device__ void LinkOctree(uint32_t N, uint32_t i, TreeCell * Octree, uint32_t * d_ChildrenCounter, uint64_t * Range, uint32_t dLevel, uint32_t TotalSize);

/*
  Computes the center of mass and total mass of the particles
  below one node. It uses the range calculated at build time
  to know exactly which ones it should access.

  Inputs:
    Index of the node.
    Pointer to the root of the octree.
    Array containing particles.
*/
__host__ __device__ void CenterOfMass(uint32_t i, TreeCell * Octree, KerParticle * Particles);

/*
  Distance between two points.
  
  Inputs:
    Points v1 and v2 as float4.

  Output:
    Distance between points as float.
*/
__host__ __device__ float Distance(float4 &v1, float4 &v2);

/*
  Computes gravitational acceleration from body v2 applied in v1,
  at distance d.

  Inputs:
    Location of particle as float4.
    Source of gravity as float4.
    Distance between them.

  Outputs:
    Acceleration as a vector float3.
*/
__host__ __device__ float3 Acceleration(float4 &v1, float4 &v2, float d);

/*
  Adds the contribution from a source in location v
  to a particle p at distance d.
*/
__host__ __device__ void AddAcceleration(KerParticle &p, float4 &v, float d);

/*
  Euler integrator, moves a particle p to the next
  position after a time step dt.
*/
__host__ __device__ void IntegrateParticle(KerParticle &p, float dt);

// Kernels

/*
  Kernel for initializing particles' initial positions
  and velocities.
*/
__global__ void KerAllocateData(uint32_t N, float4 * DataPos, float3 * DataVel, KerParticle * Particles);

/*
  Kernel that computes the size of the bounding box.
*/
__global__ void KerBoundary(float3 * Size, KerParticle * ax, KerParticle * ay, KerParticle * az, KerParticle * bx, KerParticle * by, KerParticle * bz);

/*
  Kernel that encodes particles' positions to Morton keys.
*/
__global__ void KerEncodeParticles(uint32_t N, KerParticle * Particles, uint64_t * Encoded, float3 * Size, KerParticle * MinX, KerParticle * MinY, KerParticle * MinZ, float MaxSize);

/*
  Kernel that computes the masked Morton keys at build step.
*/
__global__ void KerComputeNodes(uint32_t N, MaskedKey * Counts, uint64_t * Encoded, uint32_t k);

/*
  Kernel that counts unique nodes in octree level.
*/
__global__ void KerUniques(uint32_t CellSize, MaskedKey * Counts, uint32_t * Freq, MaskedKey * Uniques);

/*
  Kernel that sets up nodes after building an octree level.
*/
__global__ void KerInitializeOctree(uint32_t N, TreeCell * Octree, KerParticle * Particles, uint64_t * Encoded, uint32_t * Offset, MaskedKey * Uniques, uint32_t CellSize, uint32_t TotalSize, uint32_t k);

/*
  Kernel that copies from the "matrix" octree to the linear array.
*/
__global__ void KerCopyLinearOctree(uint32_t N, uint32_t dLevel, uint32_t * Offset, TreeCell ** Octree, TreeCell * LinearOctree);

/*
  Kernel that looks for children and parents of nodes, connecting the octree.
*/
__global__ void KerLinkingOctree(uint32_t N, TreeCell * Octree, uint32_t * d_ChildrenCounter, uint32_t * Range, uint32_t dLevel, uint32_t TotalSize);

/*
  Computes centers of mass of the octree.
*/
__global__ void KerCenterOfMass(uint32_t N, TreeCell * Octree, KerParticle * Particles);

/*
  Computes force contributions in the system.
*/
__global__ void KerForces(uint32_t N, KerParticle * Particles, TreeCell * Octree, uint32_t * ChildrenCounter, float3 * Size, float dt);

/*
  Copies position data to be transfered to the host.
*/
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

__host__ __device__ void MortonEncoding(uint32_t i, KerParticle * Particles, uint64_t * Encoded, float3 * Size, KerParticle * MinX, KerParticle * MinY, KerParticle * MinZ, float MaxSize)
{
  register float4 t = make_float4(0, 0, 0, 0);
  register KerParticle p = Particles[i];
  t.x = (p.Position.x + MinX->Position.x) / (Size->x + MinX->Position.x) * MaxSize;
  t.y = (p.Position.y + MinX->Position.y) / (Size->y + MinX->Position.y) * MaxSize;
  t.z = (p.Position.z + MinX->Position.z) / (Size->z + MinX->Position.z) * MaxSize;
  Encoded[i] = KerEncode(t);
}

__host__ __device__ void ComputeNodeKeys(uint32_t i, MaskedKey * Counts, uint64_t * Encoded, uint32_t k)
{
  Counts[i].key = Encoded[i] & (1i64 | Reverse((1i64 << 3 * k) - 1));
  Counts[i].index = i;
}

__host__ __device__ void LinkOctree(uint32_t N, uint32_t i, TreeCell * Octree, uint32_t * ChildrenCounter, uint32_t * Range, uint32_t dLevel, uint32_t TotalSize)
{
  register uint64_t x0, x1, x2, x3;
  register TreeCell node = Octree[i];
  if (i == 0) {
    node.Range = make_uint2(0, N);
    node.Morton = 0;
    node.Level = 0;
    node.Leaf = false;
    node.Parent = 0;
    node.Child = 1;
  } else if (node.Level > 0) {
    x0 = Range[node.Level - 1];
    x1 = Range[node.Level];
    for (uint64_t j = x0; j < x1; j++) {
      if (Octree[j].Leaf) continue;
      if ((node.Morton << 3) == Octree[j].Morton) {
#if defined( __CUDA_ARCH__ )
        atomicAdd(ChildrenCounter + j, 1);
#else
        ChildrenCounter[j]++;
#endif
        node.Parent = j;
        break;
      }
    }
    if (!node.Leaf) {
      x2 = (node.Level + 1 >= dLevel ? TotalSize : Range[node.Level + 1]);
      x3 = (node.Level + 2 >= dLevel ? TotalSize : Range[node.Level + 2]);
      for (uint32_t j = x2; j < x3; j++) {
        if ((Octree[j].Morton << 3) == node.Morton) {
          node.Child = j;
          break;
        }
      }
    }
  }
  Octree[i] = node;
}

__host__ __device__ void CenterOfMass(uint32_t i, TreeCell * Octree, KerParticle * Particles)
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
  Octree[i].Position = node.Position;
}

__host__ __device__ float Distance(float4 &v1, float4 &v2)
{
  return sqrtf(pow(v1.x - v2.x, 2) + pow(v1.y - v2.y, 2) + pow(v1.z - v2.z, 2));
}

// acceleration of v1 produced by v2, v1 -> v2.
__host__ __device__ float3 Acceleration(float4 &v1, float4 &v2, float d)
{
  float t = G_CONSTANT * v2.w / (pow(d, 3) + EPSILON);
  return make_float3(t*(v2.x - v1.x), t*(v2.y - v1.y), t*(v2.z - v1.z));
}

__host__ __device__ void AddAcceleration(KerParticle &p, float4 &v, float d)
{
  register float3 a = Acceleration(p.Position, v, d);
  p.Acceleration.x += a.x;
  p.Acceleration.y += a.y;
  p.Acceleration.z += a.z;
}

__host__ __device__ void IntegrateParticle(KerParticle &p, float dt)
{
  p.Velocity.x += p.Acceleration.x * dt;
  p.Velocity.y += p.Acceleration.y * dt;
  p.Velocity.z += p.Acceleration.z * dt;
  p.Position.x += p.Velocity.x * dt;
  p.Position.y += p.Velocity.y * dt;
  p.Position.z += p.Velocity.z * dt;
}

/***** CUDA Kernels *****/

__global__ void KerBoundary(float3 * Size, KerParticle * ax, KerParticle * ay, KerParticle * az, KerParticle * bx, KerParticle * by, KerParticle * bz)
{
  Size->x = ax->Position.x - bx->Position.x;
  Size->y = ay->Position.y - by->Position.y;
  Size->z = az->Position.z - bz->Position.z;
}

__global__ void KerAllocateData(uint32_t N, float4 * DataPos, float3 * DataVel, KerParticle * Particles)
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
  }
}

__global__ void KerEncodeParticles(uint32_t N, KerParticle * Particles, uint64_t * Encoded, float3 * Size, KerParticle * MinX, KerParticle * MinY, KerParticle * MinZ, float MaxSize)
{
  uint32_t i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < N) {
    Particles[i].Acceleration = make_float3(0, 0, 0);
    MortonEncoding(i, Particles, Encoded, Size, MinX, MinY, MinZ, MaxSize);
  }
}

__global__ void KerComputeNodes(uint32_t N, MaskedKey * Counts, uint64_t * Encoded, uint32_t k)
{
  uint32_t i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < N) { ComputeNodeKeys(i, Counts, Encoded, k); }
}

__global__ void KerUniques(uint32_t CellSizeB, uint32_t CellSizeA, MaskedKey * Counts, uint32_t * Freq, MaskedKey * Uniques)
{
  register uint32_t idx = blockIdx.x*blockDim.x + threadIdx.x;
  register uint32_t sum = 0;
  if (idx < CellSizeB) {
    for (uint32_t i = 0; i < CellSizeA; i++) {
      if (Counts[i] == Uniques[idx]) {
        sum++;
      }
    }
    Freq[idx] = sum;
  }
}

__global__ void KerInitializeOctree(uint32_t CellSizeB, uint32_t CellSizeA, TreeCell * Octree, KerParticle * Particles, uint64_t * Encoded, uint32_t * Range, MaskedKey * Cleaned, uint32_t TotalSize, uint32_t k)
{
  uint32_t i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < CellSizeB) {
    register TreeCell node = Octree[i];
    node.Range.x = Cleaned[Range[i]].index;
    node.Range.y = (i == CellSizeB - 1 ? Cleaned[CellSizeB - 1].index + 1 : Cleaned[Range[i + 1]].index);
    node.Morton = Reverse(Cleaned[Range[i]].key);
    node.Level = k;

    if (Range[i] + 1 == (i == CellSizeB - 1 ? CellSizeA : Range[i + 1])) {
      Encoded[node.Range.x] |= 1i64;
      node.Leaf = true;
    } else {
      node.Leaf = false;
    }
    Octree[i] = node;
  }
}

__global__ void KerCopyLinearOctree(uint32_t N, uint32_t dLevel, uint32_t * Offset, TreeCell ** Octree, TreeCell * LinearOctree)
{
  register uint32_t idx = threadIdx.x;
  if (idx < dLevel) {
    uint32_t end = (idx + 1 == dLevel ? N : Offset[idx + 1]);
    for (uint32_t i = Offset[idx]; i < end; i++) {
      LinearOctree[i] = Octree[idx][i-Offset[idx]];
      LinearOctree[i].Global = i;
    }
  }
}

__global__ void KerLinkingOctree(uint32_t N, TreeCell * Octree, uint32_t * ChildrenCounter, uint32_t * Offset, uint32_t dLevel, uint32_t TotalSize)
{
  uint32_t i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < TotalSize) { LinkOctree(N, i, Octree, ChildrenCounter, Offset, dLevel, TotalSize); }
}

__global__ void KerCenterOfMass(uint32_t N, TreeCell * Octree, KerParticle * Particles)
{
  uint32_t i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < N) { CenterOfMass(i, Octree, Particles); }
}

__global__ void KerForces(uint32_t N, KerParticle * Particles, TreeCell * Octree, uint32_t * ChildrenCounter, float3 * Size, float dt)
{
  uint32_t idx = blockIdx.x*blockDim.x + threadIdx.x;
  register uint32_t p0, p1, p2;
  register TreeCell localCell;
  register float d;
  register uint32_t expStop = 0, children = 0;
  register float MaxSize = max(Size->x, max(Size->y, Size->z));

  if (idx < N) {
    TreeCell currCells[N_BLOCK];
    uint32_t nextCells[N_BLOCK];
    float4 cellInt[N_BLOCK];
    register KerParticle lParticle = Particles[idx];

    p0 = 0; p1 = 8; p2 = 0;
    for (uint32_t i = 0; i < 8; i++) { nextCells[i] = i + 1; }

    while (p0 > 0 || p1 > 0 || p2 > 0) {
      // Check if there are cells in the next level stack.
      if (p1 > 0 && p0 < N_BLOCK) {
        // Move next stack to current stack.
        expStop = (p0 + p1 < N_BLOCK ? p1 : N_BLOCK - p0);
        for (uint32_t i = 0; i < expStop; i++) {
          currCells[p0 + i] = Octree[nextCells[expStop - 1 - i]];
        } p1 -= expStop; p0 += expStop;
      }

      // Check if there are cells in the current stack.
      if (p0 > 0) {
        expStop = p0; p0 = 0;
        for (uint32_t i = 0; i < expStop; i++) {
          // Move cell to register.
          localCell = currCells[i];
          // Check further exploration.
          if (localCell.Leaf) {
            // It's a leaf, add it to interactions.
            if (p2 < N_BLOCK) {
              cellInt[p2] = localCell.Position;
              p2++;
            } else {
              currCells[p0] = localCell;
              p0++;
            }
          } else {
            d = Distance(lParticle.Position, localCell.Position);
            if (MaxSize < d * (1 << localCell.Level)) {
              // Cell is too small or too far away, cut branch and add cell to interactions.
              if (p2 < N_BLOCK) {
                cellInt[p2] = localCell.Position;
                p2++;
              } else {
                currCells[p0] = localCell;
                p2++;
              }
            } else {
              // Add children to next level stack.
              children = ChildrenCounter[localCell.Global];
              if (p0 + children < N_BLOCK) {
                // If there is enough space.
                for (uint32_t i = 0; i < children; i++) {
                  nextCells[p1 + i] = localCell.Child + i;
                }
                p1 += children;
              } else {
                // If there is not, put the cell back to the front.
                currCells[p0] = localCell;
                p0++;
              }
            }
          }
        }
      }

      // Compute interactions.
      if (p2 > 0) {
        for (uint32_t i = 0; i < p2; i++) {
          AddAcceleration(lParticle, cellInt[i], Distance(lParticle.Position, cellInt[i]));
        }
        p2 = 0;
      }
    }
    
    // Once it's finished, integrate movement.
    IntegrateParticle(lParticle, dt);
    Particles[idx].Position = lParticle.Position;
  }
}

__global__ void KerUpdateData(uint32_t N, KerParticle * Particles, float4 * Data)
{
  uint32_t i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < N) { Data[i] = Particles[i].Position; }
}

NBodyKernel::~NBodyKernel()
{
  CleanGPU();
}

void NBodyKernel::CleanGPU()
{
  if (InitializedGPU) {
    delete[] TreeCellsSizes;
    delete[] OctreeCells;
    cudaFree(d_RawDataPos);
    cudaFree(d_KerParticles);
    cudaFree(d_Encoded);
    cudaFree(d_Counts);
    cudaFree(d_Size);
    cudaFree(d_OctreeCells);
  }
}

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
  if (code != cudaSuccess)
  {
    OutputDebugString(("GPUassert: " + std::string(cudaGetErrorString(code)) + " " + file + " " + std::to_string(line) + "\n").c_str());
    if (abort) exit(code);
  }
}

void NBodyKernel::InitializeGPU(uint32_t PartSize, float4* p, float3* v, float dt)
{
  if (!InitializedGPU) {
    cudaMalloc(&d_Size, sizeof(float3));
    cudaMalloc(&d_OctreeCells, sizeof(TreeCell*)*MaxDepth);
    TreeCellsSizes = new uint32_t[MaxDepth];
    OctreeCells = new TreeCell*[MaxDepth];
    benchData = new float[BENCH_AMOUNT * BENCH_FRAMES];
  }

  if (InitializedGPU && (PartSize != NParticles)) {
    delete[] RawDataPos;
    cudaFree(d_RawDataPos);
    cudaFree(d_KerParticles);
    cudaFree(d_Encoded);
    cudaFree(d_Counts);
    cudaFree(d_Offset);
    cudaFree(d_Cleaned);
    cudaFree(d_Uniques);
    cudaFree(d_Freq);
  }

  if (!InitializedGPU || (PartSize != NParticles)) {
    NParticles = PartSize;
    RawDataPos = new float4[NParticles];

    cudaMalloc(&d_Freq, sizeof(uint32_t)*NParticles);
    cudaMalloc(&d_Uniques, sizeof(MaskedKey)*NParticles);
    cudaMalloc(&d_Cleaned, sizeof(MaskedKey)*NParticles);
    cudaMalloc(&d_RawDataPos, sizeof(float4)*NParticles);
    cudaMalloc(&d_RawDataVel, sizeof(float3)*NParticles);
    cudaMalloc(&d_KerParticles, sizeof(KerParticle)*NParticles);
    cudaMalloc(&d_Encoded, sizeof(uint64_t)*NParticles);
    cudaMalloc(&d_Offset, sizeof(uint32_t)*NParticles);
    cudaMalloc(&d_Counts, sizeof(MaskedKey)*NParticles);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
  }

  deltaTime = dt;

  gpuErrchk(cudaMemcpy(d_RawDataPos, p, sizeof(float4)*NParticles, cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(d_RawDataVel, v, sizeof(float3)*NParticles, cudaMemcpyHostToDevice));

  KerAllocateData<<<(NParticles + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(NParticles, d_RawDataPos, d_RawDataVel, d_KerParticles);
  cudaFree(d_RawDataVel);
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());
  benchIt = 0;
  InitializedGPU = true;
}

void NBodyKernel::CopyPositionsToHost()
{
  if (!InitializedGPU) return;
  KerUpdateData<<<(NParticles + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(NParticles, d_KerParticles, d_RawDataPos);
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());
  cudaMemcpy(RawDataPos, d_RawDataPos, sizeof(float4)*NParticles, cudaMemcpyDeviceToHost);
}

void NBodyKernel::GPUBuildOctree()
{
  if (!InitializedGPU) return;
  thrust::device_ptr<KerParticle> dp_Particles(d_KerParticles);
  thrust::device_ptr<uint64_t> dp_Encoded(d_Encoded);
  thrust::device_ptr<MaskedKey> dp_Counts(d_Counts);
  thrust::device_ptr<MaskedKey> dp_Cleaned(d_Cleaned);
  thrust::device_ptr<MaskedKey> dp_Uniques(d_Uniques);
  thrust::device_ptr<uint32_t> dp_Freq(d_Freq);
  thrust::device_ptr<uint32_t> dp_Offset(d_Offset);
  
  if (benchIt >= BENCH_FRAMES) {
    float t = 0;
    benchFile.open(("gpubench-" + std::to_string(NParticles) + ".txt").c_str(), std::fstream::app | std::fstream::out);
    for (uint32_t i = 0; i < BENCH_FRAMES; i++, t = 0) {
      for (uint32_t j = 0; j < BENCH_AMOUNT; j++) {
        t += 0.001*benchData[i*BENCH_AMOUNT + j];
        benchFile << std::to_string(0.001*benchData[i*BENCH_AMOUNT + j]) << " ";
      }
      benchFile << std::to_string(t) << std::endl;
    }
    benchFile.close();
    benchIt = 0;
  }

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  
  /////////////////////////////
  // BOUNDING BOX
  /////////////////////////////

  cudaEventRecord(start);
  thrust::pair<thrust::device_ptr<KerParticle>, thrust::device_ptr<KerParticle>> x, y, z;
  x = thrust::minmax_element(dp_Particles, dp_Particles + NParticles, CompareParticleX());
  y = thrust::minmax_element(dp_Particles, dp_Particles + NParticles, CompareParticleY());
  z = thrust::minmax_element(dp_Particles, dp_Particles + NParticles, CompareParticleZ());

  KerBoundary<<<1, 1 >>>(d_Size, x.second.get(), x.first.get(), y.second.get(), y.first.get(), z.second.get(), z.first.get());
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());
  cudaEventRecord(stop);

  cudaEventSynchronize(stop);
  if (benchIt < BENCH_FRAMES) cudaEventElapsedTime(benchData + benchIt*BENCH_AMOUNT, start, stop);
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());

  /////////////////////////////
  // MORTON ENCODING AND SORT
  /////////////////////////////

  cudaEventRecord(start);
  KerEncodeParticles<<<(NParticles + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(NParticles, d_KerParticles, d_Encoded, d_Size, x.first.get(), y.first.get(), z.first.get(), MaxSize);
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());
  thrust::sort_by_key(dp_Encoded, dp_Encoded + NParticles, dp_Particles);

  thrust::fill(TreeCellsSizes + 1, TreeCellsSizes + MaxDepth, 0);
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());
  cudaEventRecord(stop);

  cudaEventSynchronize(stop);
  if (benchIt < BENCH_FRAMES) cudaEventElapsedTime(benchData + benchIt*BENCH_AMOUNT + 1, start, stop);

  /////////////////////////////
  // OCTREE BUILDING
  /////////////////////////////

  cudaEventRecord(start);
  uint32_t CellSizeA = 0, CellSizeB = 0, dLevel = 1;
  totalSize = 1; TreeCellsSizes[0] = 1;
  cudaMalloc(OctreeCells, sizeof(TreeCell));

  for (uint32_t k = 1; k < MaxDepth; k++) {
    KerComputeNodes<<<(NParticles + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(NParticles, d_Counts, d_Encoded, k);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    CellSizeA = thrust::remove_copy_if(dp_Counts, dp_Counts + NParticles, dp_Cleaned, MaskedKeyRemove()) - dp_Cleaned;
    //OutputDebugString(("Level: " + std::to_string(k) + " remo " + std::to_string(CellSizeA) + "\n").c_str());
    if (CellSizeA == 0) { break; }

    CellSizeB = thrust::unique_copy(dp_Cleaned, dp_Cleaned + CellSizeA, dp_Uniques) - dp_Uniques;
    //OutputDebugString(("Level: " + std::to_string(k) + " uniq " + std::to_string(CellSizeB) + "\n").c_str());
    if (CellSizeB == 0) { break; }

    dLevel++;
    TreeCellsSizes[k] = CellSizeB;
    KerUniques<<<(CellSizeB + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(CellSizeB, CellSizeA, d_Cleaned, d_Freq, d_Uniques);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    thrust::exclusive_scan(dp_Freq, dp_Freq + CellSizeB, dp_Offset);

    cudaMalloc(OctreeCells + k, sizeof(TreeCell)*CellSizeB);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    KerInitializeOctree<<<(CellSizeB + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(CellSizeB, CellSizeA, OctreeCells[k], d_KerParticles, d_Encoded, d_Offset, d_Cleaned, totalSize, k);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    totalSize += CellSizeB;
  }
  cudaEventRecord(stop);

  cudaEventSynchronize(stop);
  if (benchIt < BENCH_FRAMES) cudaEventElapsedTime(benchData + benchIt*BENCH_AMOUNT + 2, start, stop);

  /////////////////////////////
  // OCTREE LINKING
  /////////////////////////////

  cudaEventRecord(start);
  cudaMalloc(&d_FinalOctreeCells, sizeof(TreeCell)*totalSize);
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());
  uint32_t *Offset = new uint32_t[dLevel];
  thrust::exclusive_scan(TreeCellsSizes, TreeCellsSizes + dLevel, Offset);

  cudaMalloc(&d_LevelOffset, sizeof(uint32_t)*dLevel);
  cudaMemcpy(d_LevelOffset, Offset, sizeof(uint32_t)*dLevel, cudaMemcpyHostToDevice);
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());

  cudaMemcpy(d_OctreeCells, OctreeCells, sizeof(TreeCell*)*dLevel, cudaMemcpyHostToDevice);
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());

  KerCopyLinearOctree<<<1, THREADS_PER_BLOCK>>>(totalSize, dLevel, d_LevelOffset, d_OctreeCells, d_FinalOctreeCells);
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());
  for (uint32_t i = 0; i < dLevel; i++) { cudaFree(OctreeCells[i]); }

  cudaMalloc(&d_ChildrenCounter, sizeof(uint32_t)*totalSize);
  thrust::device_ptr<uint32_t> dp_ChildrenCounter(d_ChildrenCounter);
  thrust::fill(dp_ChildrenCounter, dp_ChildrenCounter + totalSize, 0);
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());

  KerLinkingOctree<<<(totalSize + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(NParticles,  d_FinalOctreeCells, d_ChildrenCounter, d_LevelOffset, dLevel, totalSize);
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());
  cudaFree(d_LevelOffset);
  delete[] Offset;
  cudaEventRecord(stop);

  cudaEventSynchronize(stop);
  if (benchIt < BENCH_FRAMES) cudaEventElapsedTime(benchData + benchIt*BENCH_AMOUNT + 3, start, stop);
  
  /////////////////////////////
  // CoM CALCULATION
  /////////////////////////////

  cudaEventRecord(start);
  KerCenterOfMass<<<(totalSize + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(totalSize, d_FinalOctreeCells, d_KerParticles);
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());
  cudaEventRecord(stop);

  cudaEventSynchronize(stop);
  if (benchIt < BENCH_FRAMES) cudaEventElapsedTime(benchData + benchIt*BENCH_AMOUNT + 4, start, stop);

  /////////////////////////////
  // FORCE CALCULATION
  /////////////////////////////

  cudaEventRecord(start);
  KerForces<<<(NParticles + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(NParticles, d_KerParticles, d_FinalOctreeCells, d_ChildrenCounter, d_Size, deltaTime);
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());
  cudaFree(d_FinalOctreeCells);
  cudaFree(d_ChildrenCounter);
  cudaEventRecord(stop);

  cudaEventSynchronize(stop);
  if (benchIt < BENCH_FRAMES) cudaEventElapsedTime(benchData + benchIt*BENCH_AMOUNT + 5, start, stop);

  cudaEventRecord(start);
  CopyPositionsToHost();
  cudaEventRecord(stop);

  cudaEventSynchronize(stop);
  if (benchIt < BENCH_FRAMES) cudaEventElapsedTime(benchData + benchIt*BENCH_AMOUNT + 6, start, stop);

  benchIt++;
}
