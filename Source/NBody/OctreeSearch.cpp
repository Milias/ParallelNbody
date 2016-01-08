// Fill out your copyright notice in the Description page of Project Settings.

#include "NBody.h"
#include "OctreeSearch.h"

// Sets default values
AOctreeSearch::AOctreeSearch() : Size(0.0), ParticleOctree(NULL), Initialized(false), ShowOctree(false), PhDeltaTime(0.01)
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

  //print(FString::SanitizeFloat(kernel(Particles.Num())));
  /*
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
  */
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

FString ToBinary(UINT64 x)
{
  FString s = "";
  for (int32 i = 63; i >= 0; i--) {
    s += ((x & (1i64 << i)) >> i ? "1" : "0");
  }
  return s;
}

void AOctreeSearch::CreateSpacePoints(int32 N, float SizeArg)
{
  Size = SizeArg;
  FVector s = FVector(SizeArg, SizeArg, SizeArg/10);
  Particles.Empty();
  Particles.SetNum(N);

  float4 *p = new float4[N];

  for (int32 i = 0; i < N; i++) {
    Particles[i].Position = i == 0 ? 0.5*s : FMath::RandPointInBox(FBox(FVector::ZeroVector, s));
    Particles[i].Velocity = i == 0 ? FVector::ZeroVector : 10 * FMath::RandRange(25.0, 50.0)*FMath::VRand();
    Particles[i].Mass = i == 0 ? 5000.0 : FMath::RandRange(1.0, 5000.0);
    p[i].x = Particles[i].Position.X;
    p[i].y = Particles[i].Position.Y;
    p[i].z = Particles[i].Position.Z;
    p[i].w = Particles[i].Mass;
  }
  Initialized = true;

  if (Ker == NULL) { Ker = new NBodyKernel(); }
  Ker->Initialize(N, p);
  Ker->GPUBuildOctree();
  Ker->CopyEncodedToHost();

  uint64_t *t = new uint64_t[N];
  bool *flags = new bool[N];
  uint64_t mask = 0;
  FString ts = "";

  print("Count: " + FString(std::to_string(Ker->thing).c_str()));

  for (int32 i = 0; i < 21; i++) {
    mask = mask | (7i64 << (3 * (20 - i) + 1));
    for (int32 j = 0; j < N; j++) {
      t[j] = Ker->GetEncoded()[j].Encoded & mask;
      if (t[j] != 0 && Particles[j].s == "") Particles[j].s = std::to_string(i).c_str();
      //print("Level: " + FString(std::to_string(i).c_str()) + ", " + FString(std::to_string(mask).c_str()));
    }
  }

  FlushPersistentDebugLines(GetWorld());
  UGameplayStatics::GetPlayerController(this, 0)->GetHUD()->RemoveAllDebugStrings();
  for (int32 i = 0; i < N; i++) {
    DrawDebugPoint(GetWorld(), Particles[i].Position, 10.0, FColor::Black, true);
    DrawDebugString(GetWorld(), Particles[i].Position, Particles[i].s, this, FColor::Red, -1.0f);
  }

  delete[] t;

  //LinearOctree p2;
  //uint64_t enc;
  //FString msg;
  //for (int32 i = 0; i < N; i++) {
    //msg = "";
    //msg += FString(std::to_string(Ker->GetEncoded()[i]).c_str());
    /*
    for (int j = 63; j >= 0; j--) {
      msg += (Ker->GetEncoded()[i] & 1i64 << j) >> j ? "1" : "0";
    }
    msg += ", ";
    enc = p2.Encode(FVector(Ker->GetParticles()[i].x, Ker->GetParticles()[i].y, Ker->GetParticles()[i].z));
    for (int j = 63; j >= 0; j--) {
      msg += (enc & 1i64 << j) >> j ? "1" : "0";
    } msg += ":  ( " + FString::SanitizeFloat(Ker->GetParticles()[i].x) + ", " + FString::SanitizeFloat(Ker->GetParticles()[i].y) + ", " + FString::SanitizeFloat(Ker->GetParticles()[i].z) + " )";
    */
    //print(msg+"\n");
  //}

  delete[] p;
}

void AOctreeSearch::CreateOctree()
{
  /*
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
  */
  //DrawDebugBox(GetWorld(), ParticleOctree->GetCenterOfMass(), FVector(Size, Size, Size), FColor::Black, true);
}

void AOctreeSearch::CleanParticles()
{
  Initialized = false;
  delete ParticleOctree;
  ParticleOctree = NULL;
  Particles.Empty();
}
