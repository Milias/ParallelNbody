// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "GameFramework/Actor.h"
#include "OctreeSearch.generated.h"

USTRUCT()
struct FParticle {
  GENERATED_USTRUCT_BODY()

  float Mass;
  FVector Position;
  FVector Velocity;
  FVector Acceleration;
  
  FParticle() : Mass(0), Position(FVector::ZeroVector), Velocity(FVector::ZeroVector), Acceleration(FVector::ZeroVector) {}
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

UCLASS()
class NBODY_API AOctreeSearch : public AActor
{
	GENERATED_BODY()
	
public:	
  float Size;
  TArray<FParticle> Particles;
  Octree* ParticleOctree;

  bool Initialized;

  UPROPERTY(BlueprintReadWrite)
  bool ShowOctree;

  UPROPERTY(BlueprintReadWrite)
  float PhDeltaTime;

	// Sets default values for this actor's properties
	AOctreeSearch();

	// Called when the game starts or when spawned
	virtual void BeginPlay() override;
	
	// Called every frame
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
