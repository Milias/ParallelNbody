// Fill out your copyright notice in the Description page of Project Settings.

using UnrealBuildTool;
using System.IO;

public class NBody : ModuleRules
{  
  private string ModulePath
  {
    get { return Path.GetDirectoryName( RulesCompiler.GetModuleFilename( this.GetType().Name ) ); }
  }
 
  private string ThirdPartyPath
  {
    get { return Path.GetFullPath( Path.Combine( ModulePath, "../../ThirdParty/" ) ); }
  }

	public NBody(TargetInfo Target)
	{
		PublicDependencyModuleNames.AddRange(new string[] { "Core", "CoreUObject", "Engine", "InputCore" });

		PrivateDependencyModuleNames.AddRange(new string[] {  });

    LoadCUDALib(Target);

		// Uncomment if you are using Slate UI
		// PrivateDependencyModuleNames.AddRange(new string[] { "Slate", "SlateCore" });
		
		// Uncomment if you are using online features
		// PrivateDependencyModuleNames.Add("OnlineSubsystem");
		// if ((Target.Platform == UnrealTargetPlatform.Win32) || (Target.Platform == UnrealTargetPlatform.Win64))
		// {
		//		if (UEBuildConfiguration.bCompileSteamOSS == true)
		//		{
		//			DynamicallyLoadedModuleNames.Add("OnlineSubsystemSteam");
		//		}
		// }
	}

  public bool LoadCUDALib(TargetInfo Target)
  {
    bool isLibrarySupported = false;

    if ((Target.Platform == UnrealTargetPlatform.Win64) || (Target.Platform == UnrealTargetPlatform.Win32))
    {
      isLibrarySupported = true;

      string LibrariesPath = Path.Combine(ThirdPartyPath, "CUDALib", "Libraries");

      PublicAdditionalLibraries.Add(Path.Combine(LibrariesPath, "CUDALib.lib"));
      PublicAdditionalLibraries.Add(Path.Combine(LibrariesPath, "cudart.lib"));
    }

    if (isLibrarySupported)
    {
      // Include path
      PublicIncludePaths.Add(Path.Combine(ThirdPartyPath, "CUDALib", "Includes"));
    }

    Definitions.Add(string.Format("WITH_CUDALIB_BINDING={0}", isLibrarySupported ? 1 : 0));

    return isLibrarySupported;
  }
}
