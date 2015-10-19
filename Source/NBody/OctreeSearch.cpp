// Fill out your copyright notice in the Description page of Project Settings.

#include "NBody.h"
#include "kernel_wrapper.h"
#include "OctreeSearch.h"

// Sets default values
AOctreeSearch::AOctreeSearch() : Size(0), ParticleOctree(NULL), Initialized(false), ShowOctree(false), PhDeltaTime(0.01)
{
 	// Set this actor to call Tick() every frame.  You can turn this off to improve performance if you don't need it.
	PrimaryActorTick.bCanEverTick = true;
}

// Called when the game starts or when spawned
void AOctreeSearch::BeginPlay()
{
  Super::BeginPlay();
}

// Called every frame
void AOctreeSearch::Tick( float DeltaTime )
{
  Super::Tick(DeltaTime);
  print(FString::SanitizeFloat(kernel(1<<20)));

  FlushPersistentDebugLines(GetWorld());
  if (PhDeltaTime > 0) {
    ComputeCubeSize();
    CreateOctree();
    for (int32 i = 0; i < Particles.Num(); i++) {
      Particles[i].Velocity += PhDeltaTime*Particles[i].Acceleration;
      Particles[i].Position += PhDeltaTime*Particles[i].Velocity;
    }
  }
  DrawOctreeBoxes(ParticleOctree);
}

void AOctreeSearch::DrawOctreeBoxes(Octree* Oct)
{
  if (Oct == NULL) { return; }
  if (Oct->IsLeafNode() && Oct->GetParticle()) {
    if(ShowOctree) DrawDebugBox(GetWorld(), Oct->GetOrigin(), FVector(Oct->GetSize(), Oct->GetSize(), Oct->GetSize()), FColor::Red, true);
    DrawDebugPoint(GetWorld(), Oct->GetParticle()->Position, 10.0, FColor::Black, true);
  } else {
    for (int32 i = 0; i < 8; i++) { DrawOctreeBoxes(Oct->GetChild(i)); }
  }
}

void AOctreeSearch::ComputeCubeSize()
{
  if (!Initialized) return;
  float t;
  Size = Particles[0].Position.GetAbsMax();
  for (int32 i = 1; i < Particles.Num(); i++) {
    t = Particles[i].Position.GetAbsMax();
    if (t > Size) { Size = t; }
  }
}

void AOctreeSearch::CreateSpacePoints(int32 N, float SizeArg)
{
  Size = SizeArg;
  FVector s = FVector(SizeArg, SizeArg, SizeArg/10);
  Particles.SetNum(N);
  for (int32 i = 0; i < N; i++) {
    Particles[i].Position = FMath::RandPointInBox(FBox(GetActorLocation() - s, GetActorLocation() + s));
    Particles[i].Velocity = 10*FMath::RandRange(25.0, 50.0)*FMath::VRand();
    Particles[i].Mass = FMath::RandRange(1.0,5000.0);
  }
  Particles[0].Position = FVector::ZeroVector;
  Particles[0].Velocity = FVector::ZeroVector;
  Particles[0].Mass = 5000.0;
  Initialized = true;
}

void AOctreeSearch::CreateOctree()
{
  if (!Initialized) return;
  FVector t = FVector::ZeroVector;
  if (ParticleOctree) { t = ParticleOctree->GetCenterOfMass(); delete ParticleOctree; }
  ParticleOctree = new Octree(t, Size);
  for (int32 i = 0; i < Particles.Num(); i++) { ParticleOctree->Add(&Particles[i]); }
  ParticleOctree->ComputeMass();

  for (int32 i = 0; i < Particles.Num(); i++) {
    Particles[i].Acceleration = FVector::ZeroVector;
    ParticleOctree->ComputeForces(&Particles[i], 1.0);
  }

  //DrawDebugBox(GetWorld(), ParticleOctree->GetCenterOfMass(), FVector(Size, Size, Size), FColor::Black, true);
}

void AOctreeSearch::CleanParticles()
{
  Initialized = false;
  delete ParticleOctree;
  ParticleOctree = NULL;
  Particles.Empty();
}
