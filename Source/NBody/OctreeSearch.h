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

USTRUCT()
struct FParticle
{
  GENERATED_USTRUCT_BODY()

  FVector Position;
  FVector Velocity;
  float Mass;
};

UCLASS()
class NBODY_API AOctreeSearch : public AActor
{
	GENERATED_BODY()

public:
  TArray<FParticle> Particles;
  NBodyKernel *Ker;

  bool Initialized;

  UPROPERTY(BlueprintReadWrite)
  float PhDeltaTime;

	AOctreeSearch();
	virtual void BeginPlay() override;
	virtual void Tick( float DeltaSeconds ) override;

  UFUNCTION(BlueprintCallable, Category = "Octree")
  void CreateSpacePoints(int32 N, float Size = 200);
  
  UFUNCTION(BlueprintCallable, Category = "Octree")
  void CleanParticles();
};