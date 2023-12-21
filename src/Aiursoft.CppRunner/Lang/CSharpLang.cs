namespace Aiursoft.CppRunner.Lang;

public class CSharpLang : ILang
{
    public string LangDisplayName { get; set; } = "C#";
    
    public string LangExtension { get; set; } = "cs";

    public string LangName { get; set; } = "cpp";

    public string DefaultCode { get; set; } = @"using System;
using System.Collections.Generic;
using System.Linq;

public class Program
{
  public static void Main()
  {
    foreach (var i in Fibonacci().Take(20))
    {
      Console.WriteLine(i);
    }
  }

  private static IEnumerable<int> Fibonacci()
  {
    int current = 1, next = 1;

    while (true) 
    {
      yield return current;
      next = current + (current = next);
    }
  }
}
";

    public string EntryFileName { get; set; } = "Program.cs";
    public string DockerImage { get; set; } = "mcr.microsoft.com/dotnet/sdk:7.0";
    public string RunCommand { get; set; } = "dotnet run --project /app/Project.csproj";

    public Dictionary<string, string> OtherFiles { get; set; } = new()
    {
        {
            "Project.csproj",
            @"<Project Sdk=""Microsoft.NET.Sdk"">
<PropertyGroup>
<OutputType>Exe</OutputType>
<TargetFramework>net7.0</TargetFramework>
<ImplicitUsings>enable</ImplicitUsings>
</PropertyGroup>
</Project>"
        }
    };
}