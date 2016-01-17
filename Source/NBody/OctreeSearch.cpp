// Fill out your copyright notice in the Description page of Project Settings.

#include "NBody.h"
#include "OctreeSearch.h"

// Sets default values
AOctreeSearch::AOctreeSearch() : Initialized(false), PhDeltaTime(0.01)
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

  if (Initialized) {
    Ker->GPUBuildOctree();
    Ker->CopyEncodedToHost();
    
    if (true) {
      for (int32 i = 0; i < Particles.Num(); i++) {
        Particles[i].Position.X = Ker->GetParticlePosition(i)->x;
        Particles[i].Position.Y = Ker->GetParticlePosition(i)->y;
        Particles[i].Position.Z = Ker->GetParticlePosition(i)->z;
      }
    }

    FlushPersistentDebugLines(GetWorld());

    UGameplayStatics::GetPlayerController(this, 0)->GetHUD()->RemoveAllDebugStrings();
    for (int32 i = 0; i < Particles.Num(); i++) {
      DrawDebugPoint(GetWorld(), Particles[i].Position, 10.0, FColor::Black, true);
    }
  }
}

void AOctreeSearch::CreateSpacePoints(int32 N, float SizeArg)
{
  if (N==2000) N = 50;
  FVector s = FVector(SizeArg, SizeArg, SizeArg/10);
  Particles.Empty();
  Particles.SetNum(N);

  float4 *p = new float4[N];
  float3 *v = new float3[N];

  for (int32 i = 0; i < N; i++) {
    Particles[i].Position = i == 0 ? 0.5*s : FMath::RandPointInBox(FBox(FVector::ZeroVector, s));
    Particles[i].Velocity = i == 0 ? FVector::ZeroVector : 10 * FMath::RandRange(25.0, 50.0)*FMath::VRand();
    Particles[i].Mass = i == 0 ? 5000.0 : FMath::RandRange(1.0, 5000.0);
    p[i].x = Particles[i].Position.X;
    p[i].y = Particles[i].Position.Y;
    p[i].z = Particles[i].Position.Z;
    p[i].w = Particles[i].Mass;
    v[i].x = Particles[i].Velocity.X;
    v[i].y = Particles[i].Velocity.Y;
    v[i].z = Particles[i].Velocity.Z;
  }
  Initialized = true;
  if (Ker == NULL) Ker = new NBodyKernel;
  Ker->InitializeGPU(N, p, v, PhDeltaTime);
  delete[] p;
  delete[] v;
}

void AOctreeSearch::CleanParticles()
{
  Initialized = false;
  Particles.Empty();
}