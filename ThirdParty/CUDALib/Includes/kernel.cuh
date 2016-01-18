#pragma once

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string>
#include <fstream>
#include <vector_types.h>

/*
  This struct encapsulates the necessary information about
  masked keys at the building step.
*/
struct MaskedKey
{
  uint64_t key;
  uint32_t index;

  __host__ __device__ MaskedKey() : key(0), index(0) {}
  __host__ __device__ MaskedKey(const MaskedKey &a) : key(a.key), index(a.index) {}

  __host__ __device__ MaskedKey& operator=(const MaskedKey &a) {
    key = a.key;
    index = a.index;
    return *this;
  }

  __host__ __device__ bool operator==(const MaskedKey &a) const {
    return key == a.key;
  }
};

struct MaskedKeyRemove
{
  __host__ __device__ bool operator()(const MaskedKey &a) {
    return a.key & 1i64;
  }
};

struct KerParticle
{
  float4 Position;
  float3 Velocity;
  float3 Acceleration;
  uint64_t Morton;

  __host__ __device__ KerParticle() : Morton(0) {}
  __host__ __device__ KerParticle& operator=(const KerParticle& p) {
    Position = p.Position;
    Velocity = p.Velocity;
    Acceleration = p.Acceleration;
    Morton = p.Morton;
    return *this;
  }
};

/*
  Comparisons for the boundary box.
*/
struct CompareParticleX
{
  __host__ __device__ bool operator()(const KerParticle &a, const KerParticle &b)
  {
    return a.Position.x < b.Position.x;
  }
};

struct CompareParticleY
{
  __host__ __device__ bool operator()(const KerParticle &a, const KerParticle &b)
  {
    return a.Position.y < b.Position.y;
  }
};

struct CompareParticleZ
{
  __host__ __device__ bool operator()(const KerParticle &a, const KerParticle &b)
  {
    return a.Position.z < b.Position.z;
  }
};

/*
  Ôctree node, internal or leaf.
*/
struct TreeCell
{
  uint2 Range;
  uint64_t Morton;
  uint32_t Level;
  bool Leaf;

  uint32_t Global, Parent, Child;
  float4 Position;

  __host__ __device__ TreeCell& TreeCell::operator=(const TreeCell& p) {
    Range = p.Range;
    Morton = p.Morton;
    Level = p.Level;
    Leaf = p.Leaf;
    Parent = p.Parent;
    Child = p.Child;
    Position = p.Position;
    return *this;
  }
};

/*
  Control class for the Kernel, just an interface to the
  CUDA implementation.

  As a general rule, d_* are pointers in device memory space.
*/

class NBodyKernel
{
private:
  uint32_t NParticles;
  float4 *RawDataPos, *d_RawDataPos;
  float3 *d_RawDataVel, *d_Size;
  KerParticle *d_KerParticles;
  uint64_t *d_Encoded;
  TreeCell **OctreeCells, **d_OctreeCells;
  TreeCell *d_FinalOctreeCells;
  MaskedKey *d_Counts, *d_Uniques, *d_Cleaned;

  bool InitializedGPU;
  float deltaTime;

  uint64_t MaxSize;
  uint32_t MaxDepth, *d_Freq, *d_Offset, *d_LevelOffset, *TreeCellsSizes, totalSize, *d_ChildrenCounter;

  std::fstream benchFile;
  float * benchData;
  uint32_t benchIt;

public:
  NBodyKernel() : InitializedGPU(false), MaxSize(1 << 20), MaxDepth(21), deltaTime(0.0), benchIt(0) {}
  ~NBodyKernel();

  void CleanGPU();
  void InitializeGPU(uint32_t PartSize, float4 *p, float3 *v, float dt);
  void CopyPositionsToHost();

  float4 * GetParticlePosition(uint32_t i) { return RawDataPos + i; }
  void GPUBuildOctree();
};
