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
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/unique.h>
#include <thrust/count.h>

#define THREADS_PER_BLOCK 256
#define G_CONSTANT 1
#define EPSILON 0.1

static inline uint32_t CLZ1(uint32_t x);
__host__ __device__ uint32_t __clz64(uint64_t x);
__host__ __device__ uint32_t __tlz64(uint64_t x);
__host__ __device__ uint64_t ReverseChar(char x);
__host__ __device__ uint64_t Reverse(uint64_t x);

__host__ __device__ uint64_t SplitBy3(uint64_t x);
__host__ __device__ uint64_t MagicBits(uint64_t x, uint64_t y, uint64_t z);
__host__ __device__ uint64_t KerEncode(float4 v);

__global__ void ParticleAlloc(uint32_t N, float4 *Data, KerParticle *Particles, uint64_t *Encoded, KerNode *Leaves);

__host__ __device__ uint2 CalculateRange(uint32_t N, KerParticle* Particles, uint64_t *Encoded, uint32_t index);
__host__ __device__ uint32_t FindSplit(uint32_t N, KerParticle* Particles, uint64_t *Encoded, uint32_t lp, uint32_t rp);
__host__ __device__ void GenerateNode(uint32_t N, KerParticle* Particles, uint64_t *Encoded, KerNode* Tree, KerNode* Leafs, uint32_t index);
__global__ void GenerateOctree(uint32_t N, KerParticle * Particles, uint64_t *Encoded, KerNode * Tree, KerNode * Leaves);

__host__ __device__ void CountNodes(uint32_t Index, uint64_t * Encoded, KerNode * Tree, uint32_t * Counts);
__global__ void GenerateOctreeNodes(uint32_t N, uint64_t * Encoded, KerNode * Tree, uint32_t * Counts);

__host__ __device__ void InitializeOctree(KerOctreeNode * Octree);
__global__ void LinkingOctreeNodes(uint32_t N, KerParticle * Particles, KerOctreeNode * Tree);

__host__ __device__ void ComputeCM(KerParticle * Particles, KerNode * Tree, uint32_t Index);
__global__ void CenterOfMassOctree(uint32_t N, KerParticle * Particles, KerNode * Tree);

__host__ __device__ float Distance(float4 &v1, float4 &v2);
__host__ __device__ float3 Acceleration(float4 &v1, float4 &v2, float d);
__host__ __device__ void ComputeAcceleration(KerParticle * Particle, KerNode * Node, float BoundingSize);
__global__ void AccelerationOctree(uint32_t N, KerParticle * Particles, KerNode * Tree, float BoundingSize);

__global__ void UpdateData(uint32_t N, KerParticle * Particles, float4 * Data);

/***** KerNode *****/

__host__ __device__ uint32_t KerNode::GetCount(uint64_t * Encoded)
{
  uint32_t t = 0;

  uint32_t deltaParent = __clz64(Encoded[Range.x] ^ Encoded[Range.y]);
  uint32_t deltaChild;

  if (LeftNode->Particle == NULL) {
    //Length of left node.
    deltaChild = __clz64(Encoded[LeftNode->Range.x] ^ Encoded[LeftNode->Range.y]);
    if (deltaChild > deltaParent) { t += deltaChild / 3 - deltaParent / 3; }
  } else {
    deltaChild = __clz64(LeftNode->Particle->Morton);
    if (deltaChild > deltaParent) { t += deltaChild / 3 - deltaParent / 3; }
  }

  if (RightNode->Particle == NULL) {
    //Length of right node.
    deltaChild = __clz64(Encoded[RightNode->Range.x] ^ Encoded[RightNode->Range.y]);
    if (deltaChild > deltaParent) { t += deltaChild / 3 - deltaParent / 3; }
  } else {
    deltaChild = __clz64(RightNode->Particle->Morton);
    if (deltaChild > deltaParent) { t += deltaChild / 3 - deltaParent / 3; }
  }

  return t;
}

/***** CUDA Kernel Functions *****/

static inline uint32_t CLZ1(uint32_t x) {
  static uint8_t const clz_lkup[] = {
    32U, 31U, 30U, 30U, 29U, 29U, 29U, 29U,
    28U, 28U, 28U, 28U, 28U, 28U, 28U, 28U,
    27U, 27U, 27U, 27U, 27U, 27U, 27U, 27U,
    27U, 27U, 27U, 27U, 27U, 27U, 27U, 27U,
    26U, 26U, 26U, 26U, 26U, 26U, 26U, 26U,
    26U, 26U, 26U, 26U, 26U, 26U, 26U, 26U,
    26U, 26U, 26U, 26U, 26U, 26U, 26U, 26U,
    26U, 26U, 26U, 26U, 26U, 26U, 26U, 26U,
    25U, 25U, 25U, 25U, 25U, 25U, 25U, 25U,
    25U, 25U, 25U, 25U, 25U, 25U, 25U, 25U,
    25U, 25U, 25U, 25U, 25U, 25U, 25U, 25U,
    25U, 25U, 25U, 25U, 25U, 25U, 25U, 25U,
    25U, 25U, 25U, 25U, 25U, 25U, 25U, 25U,
    25U, 25U, 25U, 25U, 25U, 25U, 25U, 25U,
    25U, 25U, 25U, 25U, 25U, 25U, 25U, 25U,
    25U, 25U, 25U, 25U, 25U, 25U, 25U, 25U,
    24U, 24U, 24U, 24U, 24U, 24U, 24U, 24U,
    24U, 24U, 24U, 24U, 24U, 24U, 24U, 24U,
    24U, 24U, 24U, 24U, 24U, 24U, 24U, 24U,
    24U, 24U, 24U, 24U, 24U, 24U, 24U, 24U,
    24U, 24U, 24U, 24U, 24U, 24U, 24U, 24U,
    24U, 24U, 24U, 24U, 24U, 24U, 24U, 24U,
    24U, 24U, 24U, 24U, 24U, 24U, 24U, 24U,
    24U, 24U, 24U, 24U, 24U, 24U, 24U, 24U,
    24U, 24U, 24U, 24U, 24U, 24U, 24U, 24U,
    24U, 24U, 24U, 24U, 24U, 24U, 24U, 24U,
    24U, 24U, 24U, 24U, 24U, 24U, 24U, 24U,
    24U, 24U, 24U, 24U, 24U, 24U, 24U, 24U,
    24U, 24U, 24U, 24U, 24U, 24U, 24U, 24U,
    24U, 24U, 24U, 24U, 24U, 24U, 24U, 24U,
    24U, 24U, 24U, 24U, 24U, 24U, 24U, 24U,
    24U, 24U, 24U, 24U, 24U, 24U, 24U, 24U
  };
  uint32_t n;
  if (x >= (1U << 16)) {
    if (x >= (1U << 24)) {
      n = 24U;
    } else {
      n = 16U;
    }
  } else {
    if (x >= (1U << 8)) {
      n = 8U;
    } else {
      n = 0U;
    }
  }
  return (uint32_t)clz_lkup[x >> n] - n;
}

__host__ __device__ uint32_t __clz64(uint64_t x)
{
  uint32_t t = 0;
#if defined(__CUDA_ARCH__)
  t += __clz(x >> 32);

  if (t == 32) { t += __clz(x); }
#else
  t += CLZ1(x >> 32);
  if (t == 32) { t += CLZ1(x); }
#endif
  return t;
}

__host__ __device__ uint32_t __tlz64(uint64_t x)
{
  for (uint32_t i = 0; i < 64; i++) {
    if (x & (1i64 << i)) return i;
  }
  return 64;
}

__host__ __device__ uint64_t ReverseChar(char x)
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

__global__ void ParticleAlloc(uint32_t N, float4 *Data, KerParticle *Particles, uint64_t *Encoded, KerNode *Leaves, uint32_t *Counts)
{
  uint32_t i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < N) {
    Particles[i].Position = Data[i];
    Leaves[i].Particle = Particles + i;
    Particles[i].ParentNode = Leaves + i;
    Encoded[i] = KerEncode(Particles[i].Position);
  }
  if (i + 1 < N) { Counts[i] = 0; }
}

__host__ __device__ uint2 CalculateRange(uint32_t N, KerParticle* Particles, uint64_t *Encoded, uint32_t index)
{
  if (index == 0) { return make_uint2(0, N-1); }
  int dir;
  uint32_t d_min, iind;
  iind = index;

  uint64_t ml, mc, mr;
  ml = Encoded[index - 1];
  mc = Encoded[index];
  mr = Encoded[index + 1];

  if (ml == mc && mc == mr) {
    for (; (index < N) && (Encoded[index] != Encoded[index + 1]); index++) {}
    return make_uint2(iind, index);
  } else {
    uint2 lr = make_uint2(__clz64(mc ^ ml), __clz64(mc ^ mr));
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
  while (testindex < N && (testindex >= 0 ? __clz64(mc ^ Encoded[testindex])>d_min : false)) {
    l_max *= 2;
    testindex = index + l_max*dir;
  }

  uint32_t l, t, splitPrefix; l = 0;
  int64_t nt;
  for (uint32_t div = 2; l_max / div >= 1; div *= 2) {
    t = l_max / div;
    nt = index + (l + t)*dir;
    if (nt < N) {
      splitPrefix = __clz64(mc ^ Encoded[nt]);
      if (splitPrefix > d_min) { l += t; }
    }
  }

  return dir > 0 ? make_uint2(index, index + l*dir) : make_uint2(index + l*dir, index);
}

__host__ __device__ uint32_t FindSplit(uint32_t N, KerParticle* Particles, uint64_t *Encoded, uint32_t lp, uint32_t rp)
{
  uint64_t lc, rc;
  lc = Encoded[lp];
  rc = Encoded[rp];

  if (lc == rc) return lp;

  uint64_t prefix = __clz64(lc ^ rc);
  uint32_t split = lp;
  int64_t step = rp - lp;
  int64_t ns;
  uint64_t sc, sp;
  while (step > 1) {
    step = (step + 1) >> 1;
    ns = split + step;

    if (ns < rp) {
      sc = Encoded[ns];
      sp = __clz64(lc ^ sc);
      if (sp > prefix) { split = ns; }
    }
  }

  return split;
}

__host__ __device__ void GenerateNode(uint32_t N, KerParticle* Particles, uint64_t *Encoded, KerNode* Tree, KerNode* Leaves, uint32_t index)
{
  uint2 r = CalculateRange(N, Particles, Encoded, index);
  uint32_t split = FindSplit(N, Particles, Encoded, r.x, r.y);

#if defined(__CUDA_ARCH__)
#else
  OutputDebugString(("Index: " + std::to_string(index) + ". " + std::to_string(split) + " - " + std::to_string(r.x) + ", " + std::to_string(r.y) + "\n").c_str());
#endif

  Tree[index].LeftNode = (split == r.x ? Leaves : Tree) + split;
  Tree[index].RightNode = (split + 1 == r.y ? Leaves : Tree) + split + 1;
  Tree[index].Range = r;

  Tree[index].LeftNode->ParentNode = Tree + index;
  Tree[index].RightNode->ParentNode = Tree + index;
}

__global__ void GenerateOctree(uint32_t N, KerParticle * Particles, uint64_t *Encoded, KerNode * Tree, KerNode * Leaves)
{
  uint32_t i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < N) { GenerateNode(N, Particles, Encoded, Tree, Leaves, i); }
}

__host__ __device__ void CountNodes(uint32_t Index, uint64_t * Encoded, KerNode * Tree, uint32_t * Counts)
{
  Counts[Index] += Tree[Index].GetCount(Encoded);
}

__global__ void GenerateOctreeNodes(uint32_t N, uint64_t * Encoded, KerNode * Tree, uint32_t * Counts)
{
  uint32_t i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i + 1 < N) { CountNodes(i, Encoded, Tree, Counts); }
}

__host__ __device__ void InitializeOctree(uint32_t Index, KerOctreeNode * Octree, uint64_t * Encoded, uint32_t * Counts, KerNode * Tree)
{
#if defined(__CUDA_ARCH__)
#else
  OutputDebugString(("Index: " + std::to_string(__clz64(Encoded[Tree[Index].Range.x] ^ Encoded[Tree[Index].Range.y])) + "\n").c_str());
#endif
  uint32_t start = Index == 0 ? 0 : Counts[Index - 1];
  for (uint32_t i = start; i < Counts[Index]; i++) {
    Octree[i].RadixParent = Tree + i;
    Octree[i].Morton = Encoded[Tree[Index].Range.x] & Reverse((1i64 << (__clz64(Encoded[Tree[Index].Range.x] ^ Encoded[Tree[Index].Range.y]) + 1)) - 1);
  }
}

__global__ void LinkingOctreeNodes(uint32_t N, KerParticle * Particles, KerOctreeNode * Octree)
{
  uint32_t i = blockIdx.x*blockDim.x + threadIdx.x;
//  if (i < N) { InitializeOctree(i, Octree); }
}

__host__ __device__ void ComputeCM(KerParticle * Particles, KerNode * Tree, uint32_t Index)
{
  Tree[Index].CenterOfMass = make_float4(0, 0, 0, 0);
  for (uint32_t i = Tree[Index].Range.x; i < Tree[Index].Range.y; i++) {
    Tree[Index].CenterOfMass.x += Particles[i].Position.w * Particles[i].Position.x;
    Tree[Index].CenterOfMass.y += Particles[i].Position.w * Particles[i].Position.y;
    Tree[Index].CenterOfMass.z += Particles[i].Position.w * Particles[i].Position.z;
    Tree[Index].CenterOfMass.w += Particles[i].Position.w;
  }
  Tree[Index].CenterOfMass.x /= Tree[Index].CenterOfMass.w;
  Tree[Index].CenterOfMass.y /= Tree[Index].CenterOfMass.w;
  Tree[Index].CenterOfMass.z /= Tree[Index].CenterOfMass.w;
}

__global__ void CenterOfMassOctree(uint32_t N, KerParticle * Particles, KerNode * Tree)
{
  uint32_t i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i + 1 < N) { ComputeCM(Particles, Tree, i); }
}

__host__ __device__ float Distance(float4 &v1, float4 &v2)
{
  return sqrtf(pow(v1.x - v2.x, 2) + pow(v1.y - v2.y, 2) + pow(v1.z - v2.z, 2));
}

// acceleration of v1 produced by v2, v1 -> v2.
__host__ __device__ float3 Acceleration(float4 &v1, float4 &v2, float d)
{
  float t = G_CONSTANT * v2.w / pow(d+EPSILON, 3);
  return make_float3(t*(v2.x - v1.x), t*(v2.y - v1.y), t*(v2.z - v1.z));
}

__host__ __device__ void ComputeAcceleration(KerParticle * Particle, KerNode * Node, float BoundingSize)
{
  float d = 0.0;
  if (Node->Particle == NULL) {
    d = Distance(Node->LeftNode->CenterOfMass, Particle->Position);
    if (BoundingSize / d < 1) {
      float3 a = Acceleration(Particle->Position, Node->LeftNode->CenterOfMass, d);
      Particle->Acceleration.x += a.x;
      Particle->Acceleration.y += a.y;
      Particle->Acceleration.z += a.z;
    } else { ComputeAcceleration(Particle, Node->LeftNode, BoundingSize / 2); }

    d = Distance(Node->RightNode->CenterOfMass, Particle->Position);
    if (BoundingSize / d < 1) {
      float3 a = Acceleration(Particle->Position, Node->RightNode->CenterOfMass, d);
      Particle->Acceleration.x += a.x;
      Particle->Acceleration.y += a.y;
      Particle->Acceleration.z += a.z;
    } else { ComputeAcceleration(Particle, Node->RightNode, BoundingSize / 2); }
  } else {
    float3 a = Acceleration(Particle->Position, Node->RightNode->CenterOfMass, Distance(Particle->Position, Node->Particle->Position));
    Particle->Acceleration.x += a.x;
    Particle->Acceleration.y += a.y;
    Particle->Acceleration.z += a.z;
  }
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
    cudaFree(d_Encoded);
    cudaFree(d_KerTree);
    cudaFree(d_KerLeaves);
    cudaFree(d_Counts);
    delete[] RawData;
    delete[] KerParticles;
    delete[] KerTree;
    delete[] KerLeaves;
    delete[] Counts;
    delete[] Encoded;
  }
}

void NBodyKernel::Initialize(uint32_t PartSize, float4* PartArray)
{
  if (Initialized && (PartSize != NParticles)) {
    cudaFree(d_RawData);
    cudaFree(d_KerParticles);
    cudaFree(d_Encoded);
    cudaFree(d_KerTree);
    cudaFree(d_KerLeaves);
    cudaFree(d_Counts);
    delete[] RawData;
    delete[] KerParticles;
    delete[] KerTree;
    delete[] KerLeaves;
    delete[] Counts;
    delete[] Encoded;
  }

  if (!Initialized || (PartSize != NParticles)) {
    NParticles = PartSize;
    KerParticles = new KerParticle[NParticles];
    RawData = new float4[NParticles];
    KerTree = new KerNode[NParticles - 1];
    KerLeaves = new KerNode[NParticles];
    Counts = new uint32_t[NParticles];
    Encoded = new uint64_t[NParticles];

    cudaMalloc(&d_RawData, sizeof(float4)*NParticles);
    cudaMalloc(&d_KerParticles, sizeof(KerParticle)*NParticles);
    cudaMalloc(&d_Encoded, sizeof(uint64_t)*NParticles);
    cudaMalloc(&d_KerLeaves, sizeof(KerNode)*NParticles);
    cudaMalloc(&d_KerTree, sizeof(KerNode)*(NParticles - 1));
    cudaMalloc(&d_Counts, sizeof(uint32_t)*(NParticles));
  }

  for (uint32_t i = 0; i < NParticles; i++) { RawData[i] = PartArray[i]; }
  //cudaMemcpy(d_RawData, PartArray, sizeof(float4)*NParticles, cudaMemcpyHostToDevice);

  //ParticleAlloc<<<(NParticles + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(NParticles, d_RawData, d_KerParticles, d_Encoded, d_KerLeaves, d_Counts);
  Initialized = true;
}

void NBodyKernel::CopyEncodedToHost()
{
  if (!Initialized) return;
  UpdateData<<<(NParticles + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(NParticles, d_KerParticles, d_RawData);
  cudaMemcpy(RawData, d_RawData, sizeof(float4)*NParticles, cudaMemcpyDeviceToHost);
}

void NBodyKernel::GPUBuildOctree()
{
  if (!Initialized) return;
  thrust::device_ptr<KerParticle> p(d_KerParticles);
  thrust::device_ptr<uint64_t> e(d_Encoded);
  thrust::device_ptr<uint32_t> c(d_Counts);

  thrust::sort_by_key(e, e + NParticles, p);
  GenerateOctree<<<(NParticles + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(NParticles, d_KerParticles, d_Encoded, d_KerTree, d_KerLeaves);
  GenerateOctreeNodes<<<(NParticles + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(NParticles, d_Encoded, d_KerTree, d_Counts);

  thrust::inclusive_scan(c, c + NParticles - 1, c);

  uint32_t octreeSize;
  cudaMemcpy(&octreeSize, d_Counts + NParticles - 2, sizeof(uint32_t), cudaMemcpyDeviceToHost);

  if (d_LinearOctree == NULL) { cudaMalloc(&d_LinearOctree, sizeof(KerOctreeNode)*octreeSize); }

  cudaMemcpy(Counts, d_KerTree, sizeof(KerNode)*(NParticles - 1), cudaMemcpyDeviceToHost);

  //CenterOfMassOctree<<<(NParticles + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(NParticles, d_KerParticles, d_KerTree);
}

void NBodyKernel::CPUBuildOctree()
{
  if (!Initialized) return;
  for (uint32_t i = 0; i < NParticles; i++) {
    KerParticles[i].Position = RawData[i];
    KerLeaves[i].Particle = KerParticles + i;
    KerParticles[i].ParentNode = KerLeaves + i;
    Encoded[i] = KerEncode(KerParticles[i].Position);
    /*if (i + 1 < NParticles)*/ Counts[i] = 0;
  }

  thrust::sort_by_key(Encoded, Encoded + NParticles, KerParticles);

  char msg[65]; msg[64] = '\0';
  for (uint32_t i = 0; i < NParticles; i++) {
    for (int j = 0; j < 64; j++) { msg[j] = (Encoded[i] & 1i64 << j) ? '1' : '0'; }
    OutputDebugString(("Index: " + std::to_string(i) + ", \t" + msg + "\n").c_str());
  }

  TreeCell ** TC = new TreeCell*[21];
  uint32_t * values = new uint32_t[NParticles];
  uint32_t * temp, *temp2;
  temp2 = new uint32_t[NParticles];
  uint32_t t2, TCcount = 0;
  bool finish = false;
  for (uint32_t k = 1; k < 21; k++, TCcount++) {
    finish = true;
    for (uint32_t i = 0; i < NParticles; i++) {
      for (int j = 0; j < 64; j++) { msg[j] = (Encoded[i] & 1i64 << j) ? '1' : '0'; }
      OutputDebugString(("Index: " + std::to_string(i) + ", \t" + msg + "\n").c_str());
      if ((Encoded[i] & 1i64) == 0) { finish = false; }
      Counts[i] = ((Encoded[i] & Reverse((1i64 << (3 * k)) - 1)) >> (64 - 3 * k));
    }
    if (finish) break;

    for (uint32_t i = 0; i < NParticles; i++) { OutputDebugString((std::to_string(Counts[i]) + ", ").c_str()); }
    OutputDebugString("\n\n");
    temp = thrust::unique_copy(Counts, Counts + NParticles, values);
    t2 = temp - values;
    temp = new uint32_t[t2];
    for (uint32_t i = 0; i < t2; i++) { temp[i] = thrust::count(Counts, Counts + NParticles, values[i]); }
    thrust::exclusive_scan(temp, temp + t2, temp2);
    for (uint32_t i = 0; i < t2; i++) { OutputDebugString((std::to_string(temp2[i]) + ", ").c_str()); }
    OutputDebugString("\n\n");
    TC[k] = new TreeCell[t2];
    for (uint32_t i = 0; i < t2; i++) {
      TC[k][i].Range = make_uint2(temp2[i], i == t2 - 1 ? t2 : temp2[i + 1]);
      if (TC[k][i].Range.x + 1 == TC[k][i].Range.y) {
        Encoded[TC[k][i].Range.x] |= 1i64;
      }
      OutputDebugString(("[" + std::to_string(TC[k][i].Range.x) + ", " + std::to_string(TC[k][i].Range.y) + "]\n").c_str());
    }
    //OutputDebugString("\n\n");
    delete[] temp;
    //delete[] temp2;
  }
  if (TCcount != 20) delete[] temp;
  for (uint32_t i = 0; i < TCcount; i++) {
    delete[] TC[i];
  }

  /*
  for (uint32_t i = 0; i < NParticles; i++) { KerParticles[i].Morton = Encoded[i]; }
  for (uint32_t i = 0; i < NParticles; i++) { GenerateNode(NParticles, KerParticles, Encoded, KerTree, KerLeaves, i); }
  for (uint32_t i = 0; i + 1 < NParticles; i++) { CountNodes(i, Encoded, KerTree, Counts); }
  */
  /*OutputDebugString("\n\n");
  for (uint32_t i = 0; i + 1 < NParticles; i++) { OutputDebugString((std::to_string(Counts[i]) + ", ").c_str()); }
  OutputDebugString("\n\n");*/
  //thrust::inclusive_scan(Counts, Counts + NParticles - 1, Counts);
  //uint32_t Size = thrust::reduce(Counts, Counts + NParticles - 1) + 1;
  //thrust::inclusive_scan(Counts, Counts + NParticles - 1, Counts);
  /*for (uint32_t i = 0; i + 1 < NParticles; i++) { OutputDebugString((std::to_string(Counts[i]) + ", ").c_str()); }
  OutputDebugString("\n\n");*/
  /*
  LinearOctree = new KerOctreeNode[Size];
  //thrust::inclusive_scan(Counts, Counts + NParticles - 1, Counts);
  OutputDebugString((std::to_string(Size) + " \n").c_str());
  for (uint32_t i = 0; i + 1 < NParticles; i++) { OutputDebugString((std::to_string(Counts[i]) + ", ").c_str()); }
  OutputDebugString("\n\n");
  for (uint32_t i = 0; i + 1 < NParticles; i++) { InitializeOctree(i, LinearOctree, Encoded, Counts, KerTree); }
  for (uint32_t i = 0; i < Size; i++) {
    for (int j = 0; j < 64; j++) { msg[j] = (LinearOctree[i].Morton & 1i64 << j) ? '1' : '0'; }
    OutputDebugString(("Index: " + std::to_string(i) + ", \t" + msg + "\n").c_str());
  }

  OutputDebugString("\n\n");
  */
}
