#pragma once

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <vector_types.h>

struct KerParticle;
struct KerNode;

struct KerNode
{
  KerNode * LeftNode;
  KerNode * RightNode;

  KerParticle * Particle;
  KerNode * ParentNode;
  float4 CenterOfMass;

  uint2 Range;

  __host__ __device__ KerNode() : LeftNode(NULL), RightNode(NULL), Particle(NULL), ParentNode(NULL) {}
  __host__ __device__ uint32_t GetCount(uint64_t * Encoded);
};

struct KerParticle
{
  float4 Position;
  float3 Acceleration;
  KerNode * ParentNode;
  uint64_t Morton;

  __host__ __device__ KerParticle() : ParentNode(NULL) {}
  __host__ __device__ KerParticle& operator=(const KerParticle& p) {
    Position = p.Position;
    Acceleration = p.Acceleration;
    ParentNode = p.ParentNode;
    Morton = p.Morton;
    return *this;
  }
};

struct KerOctreeNode
{
  float4 CenterOfMass;
  uint64_t Morton;
  uint32_t Pointer;
  KerParticle * Particle;
  KerOctreeNode ** Children;
  KerOctreeNode * Parent;
  KerNode * RadixParent;
  uint8_t Level;

  __host__ __device__ KerOctreeNode() : Morton(0), Pointer(0), Particle(NULL), Children(NULL), Parent(NULL), RadixParent(NULL), Level(0) {}

  __host__ __device__ ~KerOctreeNode() {
    if (Children != NULL) { delete[] Children; }
  }

  __host__ __device__ KerOctreeNode& operator=(const KerOctreeNode& p) {
    CenterOfMass = p.CenterOfMass;
    Morton = p.Morton;
    Children = p.Children;
    Parent = p.Parent;
    Level = p.Level;
    return *this;
  }

  __host__ __device__ void CreateChildren() {
    Children = new KerOctreeNode*[8];
  }

  __host__ __device__ void SetChild(KerOctreeNode *p) {
    if (Pointer >= 8) return;
    Children[Pointer] = p; Pointer++;
  }
};

struct TreeCell
{
  uint2 Range;
};

class NBodyKernel
{
private:
  uint32_t NParticles;

public:
  float4 *RawData, *d_RawData;
  KerParticle *KerParticles, *d_KerParticles;
  uint64_t *Encoded, *d_Encoded;
  KerNode *KerTree, *KerLeaves, *d_KerTree, *d_KerLeaves;
  KerOctreeNode *LinearOctree, *d_LinearOctree;

  bool Initialized;

  uint64_t MaxSize;
  uint32_t MaxDepth, *Counts, *d_Counts;

  NBodyKernel() : Initialized(false), MaxSize(1<<20), MaxDepth(21) {}
  ~NBodyKernel();

  void Initialize(uint32_t PartSize, float4* PartArray);
  void CopyEncodedToHost();

  void GPUBuildOctree();
  void CPUBuildOctree();

  KerParticle* GetEncoded() { return KerParticles; }
};
