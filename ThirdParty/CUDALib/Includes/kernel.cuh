#pragma once

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <vector_types.h>

struct KerParticle;
struct KerNode;

struct KerNode
{
  KerNode* LeftNode;
  KerNode* RightNode;

  KerParticle* Particle;
  KerNode* ParentNode;

  KerNode() : LeftNode(NULL), RightNode(NULL), Particle(NULL), ParentNode(NULL) {}
};

struct KerParticle
{
  float4 Position;
  uint64_t Encoded;
  KerNode* ParentNode;

  KerParticle() {}
};

class NBodyKernel
{
private:
  uint32_t NParticles;

  float4 * RawData;
  float4 * d_RawData;
  KerParticle* KerParticles;
  KerParticle* d_KerParticles;
  KerNode* d_KerTree;
  KerNode* d_KerLeaves;

  bool Initialized;

  uint64_t MaxSize;
  uint32_t MaxDepth;

public:

  uint64_t *blockSum;

  int thing;

  NBodyKernel() : Initialized(false), MaxSize(1<<20), MaxDepth(21) {
    /*for (uint32_t i = 1; i < MaxDepth; i++) {
      LevelMasks[i] = LevelMasks[i-1] | (7i64 << (3 * (MaxDepth - 1 - i) + 1));
    }*/
  }
  ~NBodyKernel();

  void Initialize(uint32_t PartSize, float4* PartArray);
  void CopyEncodedToHost();

  void GPUBuildOctree();

  KerParticle* GetEncoded() { return KerParticles; }
};
