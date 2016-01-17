#pragma once

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <vector_types.h>

struct KerParticle
{
  float4 Position;
  float3 Velocity;
  float3 Acceleration;
  uint64_t Morton;
  uint32_t Parent;

  __host__ __device__ KerParticle() : Morton(0), Parent(0) {}
  __host__ __device__ KerParticle& operator=(const KerParticle& p) {
    Position = p.Position;
    Velocity = p.Velocity;
    Acceleration = p.Acceleration;
    Morton = p.Morton;
    Parent = p.Parent;
    return *this;
  }
};

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

struct TreeCell
{
  uint2 Range;
  uint64_t Morton;
  uint32_t Level;
  bool Leaf;

  uint32_t Parent;
  uint32_t Child;
  KerParticle * Particle;

  float4 Position;

  __host__ __device__ TreeCell() : Morton(0), Level(0), Leaf(false), Parent(NULL), Child(NULL), Particle(NULL) {}
  __host__ __device__ ~TreeCell() {}
  __host__ __device__ TreeCell& TreeCell::operator=(const TreeCell& p) {
    Range = p.Range;
    Morton = p.Morton;
    Level = p.Level;
    Leaf = p.Leaf;
    Parent = p.Parent;
    Child = p.Child;
    Particle = p.Particle;
    Position = p.Position;
    return *this;
  }
};

class NBodyKernel
{
private:
  uint32_t NParticles;

public:
  float4 *RawDataPos, *d_RawDataPos;
  float3 *RawDataVel, *d_RawDataVel;
  KerParticle *KerParticles, *d_KerParticles;
  uint64_t *Encoded, *d_Encoded, *Counts, *d_Counts, *values, *d_values, *temp64, *d_temp64;
  TreeCell **OctreeCells, **d_OctreeCells;
  TreeCell *FinalOctreeCells, *d_FinalOctreeCells;

  float3 CubeCornerA, CubeCornerB, Size, *d_Size;
  bool InitializedCPU, InitializedGPU;
  float deltaTime;

  uint64_t MaxSize;
  uint32_t MaxDepth, *Temp, *d_Temp, *TreeCellsSizes, totalSize, *ChildrenCounter, *d_ChildrenCounter;

  NBodyKernel() : InitializedCPU(false), InitializedGPU(false), MaxSize(1 << 20), MaxDepth(21), deltaTime(0) {}
  ~NBodyKernel();

  void CleanCPU();
  void CleanGPU();
  void InitializeCPU(uint32_t PartSize, float4 *p, float3 *v, float dt);
  void InitializeGPU(uint32_t PartSize, float4 *p, float3 *v, float dt);
  void CopyEncodedToHost();

  float4 * GetParticlePosition(uint32_t i) {
    if (InitializedCPU) {
      return &(KerParticles[i].Position);
    } else if (InitializedGPU) {
      return RawDataPos + i;
    }
    return NULL;
  }

  void GPUBuildOctree();
  void CPUBuildOctree();

  KerParticle* GetEncoded() { return KerParticles; }
};
