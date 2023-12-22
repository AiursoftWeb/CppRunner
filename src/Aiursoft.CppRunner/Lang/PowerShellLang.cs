namespace Aiursoft.CppRunner.Lang;

public class PowerShellLang : ILang
{
    public string LangDisplayName { get; set; } = "PowerShell Core";

    public string LangExtension { get; set; } = "ps1";

    public string LangName { get; set; } = "powershell";

    public string DefaultCode { get; set; } =
        """
        function Get-Fibonacci {
            $current = 1
            $next = 1
            while ($true) {
                $current
                $temp = $current
                $current = $next
                $next = $temp + $next
            }
        }
        
        Get-Fibonacci | Select-Object -First 20
        
        """;

    public string EntryFileName { get; set; } = "main.ps1";

    public string DockerImage { get; set; } = "mcr.microsoft.com/powershell";

    public string RunCommand { get; set; } = "pwsh /app/main.ps1";

    public Dictionary<string, string> OtherFiles { get; set; } = new();
}