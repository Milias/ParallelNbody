// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include <functional>
#undef NDEBUG
#include <assert.h>
#include <string>
#include <cstdint>
#include "GameFramework/Actor.h"
#include "kernel_wrapper.h"
#include "OctreeSearch.generated.h"

typedef unsigned long long uint64_t;

USTRUCT()
struct FParticle {
  GENERATED_USTRUCT_BODY()

  float Mass;
  FVector Position;
  FVector Velocity;
  FVector Acceleration;

  FString s;

  FParticle() : Mass(0), Position(FVector::ZeroVector), Velocity(FVector::ZeroVector), Acceleration(FVector::ZeroVector), s("") {}
};

class Octree
{
private:
  FParticle* Particle;
  FVector Origin;
  float Size;
  float TotalMass;
  FVector CenterOfMass;
  float Epsilon;

  Octree *Children[8];

public:
  Octree(const FVector& Origin, float Size) : Particle(NULL), Origin(Origin), Size(Size), TotalMass(0), CenterOfMass(FVector::ZeroVector), Epsilon(0.1) {
    for (int32 i = 0; i < 8; i++) { Children[i] = NULL; }
  }

  ~Octree() {
    if (!IsLeafNode()) {
      for (int32 i = 0; i < 8; i++) { delete Children[i]; }
    }
  }

  FParticle* GetParticle() const { return Particle; }
  FVector GetOrigin() const { return Origin; }
  float GetSize() const { return Size; }
  float GetTotalMass() const { return TotalMass; }
  FVector GetCenterOfMass() const { return CenterOfMass; }
  Octree* GetChild(int32 i) { return Children[i]; }

  int32 GetOctant(const FVector& Point) {
    int32 octant = 0;
    if (Point.X >= Origin.X) octant |= 4;
    if (Point.Y >= Origin.Y) octant |= 2;
    if (Point.Z >= Origin.Z) octant |= 1;
    return octant;
  }

  bool IsLeafNode() const { return Children[0] == NULL; }

  void Add(FParticle* Particle) {
    if (IsLeafNode()) {
      if (this->Particle == NULL) {
        this->Particle = Particle;
      } else {
        FParticle* old = this->Particle;
        this->Particle = NULL;

        FVector center;
        for (int32 i = 0; i < 8; i++) {
          center = Origin;
          center.X += Size * (i & 4 ? 0.5 : -0.5);
          center.Y += Size * (i & 2 ? 0.5 : -0.5);
          center.Z += Size * (i & 1 ? 0.5 : -0.5);
          Children[i] = new Octree(center, 0.5*Size);
        }

        Children[GetOctant(old->Position)]->Add(old);
        Children[GetOctant(Particle->Position)]->Add(Particle);
      }
    } else { Children[GetOctant(Particle->Position)]->Add(Particle); }
  }

  void ComputeMass() {
    if (IsLeafNode()) {
      if (Particle) {
        CenterOfMass = Particle->Position;
        TotalMass = Particle->Mass;
      }
    } else {
      for (int32 i = 0; i < 8; i++) {
        Children[i]->ComputeMass();
        TotalMass += Children[i]->TotalMass;
        CenterOfMass += Children[i]->TotalMass*Children[i]->CenterOfMass;
      }
      if (TotalMass) { CenterOfMass /= TotalMass; } else { CenterOfMass = Origin; }
    }
  }

  void ComputeForces(FParticle* Particle, float Theta) {
    if (IsLeafNode() && this->Particle == NULL) { return; }
    float d = FVector::Dist(CenterOfMass, Particle->Position) + Epsilon;
    if (Particle == this->Particle) { return; }
    if (Size / d < Theta || this->Particle) {
      Particle->Acceleration += 1e4*TotalMass/pow(d,3) * (CenterOfMass - Particle->Position);
    } else if(!IsLeafNode()) {
      for (int32 i = 0; i < 8; i++) { Children[i]->ComputeForces(Particle, Theta); }
    }
  }
};

struct LinearCell {
  FParticle* Particle;
  uint64_t Origin;
  char Level;
  float Size;
  float TotalMass;
  FVector CenterOfMass;
  float Epsilon;

  LinearCell *Parent;
  LinearCell *Children[8];

  LinearCell(LinearCell* Parent, char Level, uint64_t Origin, float Size, float Epsilon = 0.1) : Parent(Parent), Level(Level), Origin(Origin), Size(Size), Epsilon(Epsilon) {
    assert(Level < 21 && Level >= 0);
  }

  ~LinearCell() {
    if (Children[0] != NULL) {
      for (char i = 0; i < 8; i++) { delete Children[i]; }
    }
  }

  bool IsLeafNode() { return Children[0] == NULL; }
};

class LinearOctree
{
private:
  char MaxDepth;
  float MaxSize;

  LinearCell **Cells;

  uint64_t SplitBy3(uint64_t x) {
    x &= 0x1fffff;
    x = (x | (x <<  32)) & 0x1f00000000ffff;
    x = (x | (x <<  16)) & 0x1f0000ff0000ff;
    x = (x | (x << 8)) & 0x100f00f00f00f00f;
    x = (x | (x << 4)) & 0x10c30c30c30c30c3;
    x = (x | (x << 2)) & 0x1249249249249249;
    return x;
  }

  uint64_t MagicBits(uint64_t x, uint64_t y, uint64_t z) {
    return SplitBy3(x) | SplitBy3(y) << 1 | SplitBy3(z) << 2;
  }

public:
  TArray<FParticle*> Particles;
  LinearOctree() : MaxDepth(21), MaxSize(1 << 20) {}
  ~LinearOctree() {}

  uint64_t Encode(FVector p) {
    //Assuming p.i in [0,1)
    uint64_t x, y, z;
    x = (uint64_t)p.X;
    y = (uint64_t)p.Y;
    z = (uint64_t)p.Z;
    return MagicBits(x, y, z);
  }

  LinearCell* CreateChildren(LinearCell* Parent) {
    assert(Parent->Level + 1 < MaxDepth);
    uint64_t Anchor;
    for (int32 i = 0; i < 8; i++) {
      Anchor = Parent->Origin | (i << 3 * (Parent->Level + 1));
      Parent->Children[i] = new LinearCell(Parent, Parent->Level + 1, Anchor, Parent->Size * 0.5, Parent->Epsilon);
    }
    return Parent->Children[0];
  }

  void AllocateParticles(TArray<FParticle>& ParticleArray)
  {
    Particles.SetNum(ParticleArray.Num());
    for (uint64_t j = 0; j < Particles.Num(); j++) {
      Particles[j] = &ParticleArray[j];
    }
  }

  void SortParticles()
  {
    bool *e = new bool[Particles.Num()];
    uint64_t *f = new uint64_t[Particles.Num()];
    
    for (uint64_t j = 0; j < Particles.Num(); j++) {
      e[j] = false;
      f[j] = 0;
    }
    
    TArray<FParticle*> SortedParticles; SortedParticles.SetNum(Particles.Num());
    uint64_t totalFalses;
    for (uint64_t i = 0; i < 64; i++) {
      for (uint64_t j = 0; j < Particles.Num(); j++) {
        e[j] = ((Encode(Particles[j]->Position) & 1i64 << i) >> i) == 0;
        if (j > 0) f[j] = (e[j-1] ? 1 : 0) + f[j - 1];
      }
      totalFalses = (e[Particles.Num() - 1] ? 1 : 0) + f[Particles.Num() - 1];
      for (uint64_t j = 0; j < Particles.Num(); j++) {
        SortedParticles[e[j] ? f[j] : j - f[j] + totalFalses] = Particles[j];
      }
      for (uint64_t j = 0; j < Particles.Num(); j++) {
        Particles[j] = SortedParticles[j];
      }
    }

    delete[] e; delete[] f;
  }
  
  LinearCell* Traverse(LinearCell* UCell, uint64_t Loc) {
    LinearCell* cell = UCell;
    char Level = UCell->Level - 1;
    uint64_t childIndex;
    while (cell->Children) {
      childIndex = (Loc & (7 << 3 * Level)) >> 3 * Level;
      cell = cell->Children[childIndex];
    }
    return cell;
  }
};

UCLASS()
class NBODY_API AOctreeSearch : public AActor
{
	GENERATED_BODY()

public:
  float Size;
  TArray<FParticle> Particles;
  Octree *ParticleOctree;
  LinearOctree *LinOctree;

  NBodyKernel *Ker;

  bool Initialized;

  UPROPERTY(BlueprintReadWrite)
  bool ShowOctree;

  UPROPERTY(BlueprintReadWrite)
  float PhDeltaTime;

	AOctreeSearch();
	virtual void BeginPlay() override;
	virtual void Tick( float DeltaSeconds ) override;

  void DrawOctreeBoxes(Octree* Oct);
  void ComputeCubeSize();

  UFUNCTION(BlueprintCallable, Category = "Octree")
  void CreateSpacePoints(int32 N, float Size = 200);

  UFUNCTION(BlueprintCallable, Category = "Octree")
  void CreateOctree();

  UFUNCTION(BlueprintCallable, Category = "Octree")
  void CleanParticles();
};