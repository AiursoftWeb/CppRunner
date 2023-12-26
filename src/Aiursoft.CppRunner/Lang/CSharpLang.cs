namespace Aiursoft.CppRunner.Lang;

public class CSharpLang : ILang
{
    public string LangDisplayName => "C# (.NET 7.0)";

    public string LangExtension => "cs";

    public string LangName => "csharp";

    public string DefaultCode =>
        """
        using System;
        using System.Collections.Generic;
        using System.Linq;

        public class Program
        {
            private static IEnumerable<int> Fibonacci()
            {
                int current = 1, next = 1;
        
                while (true)
                {
                    yield return current;
                    next = current + (current = next);
                }
            }
        
            public static void Main()
            {
                foreach (var i in Fibonacci().Take(20))
                {
                    Console.WriteLine(i);
                }
            }
        }

        """;

    public string EntryFileName => "Program.cs";
    public string DockerImage => "mcr.microsoft.com/dotnet/sdk:7.0";
    public string RunCommand => "dotnet run --project /app/Project.csproj";

    public Dictionary<string, string> OtherFiles => new()
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