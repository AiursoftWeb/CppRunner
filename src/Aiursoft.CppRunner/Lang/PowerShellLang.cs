namespace Aiursoft.CppRunner.Lang;

public class PowerShellLang : ILang
{
    public string LangDisplayName { get; set; } = "PowerShell (PowerShell 7.1)";

    public string LangExtension { get; set; } = "ps1";

    public string LangName { get; set; } = "powershell";

    public string DefaultCode { get; set; } =
        """
        function Get-Fibonacci {
            $current, $next = 0, 1
            while ($true) {
                $current
                $current, $next = $next, $current + $next
            }
        }

        Get-Fibonacci | Select-Object -First 20
        """;

    public string EntryFileName { get; set; } = "main.ps1";

    public string DockerImage { get; set; } = "mcr.microsoft.com/powershell:7.1-alpine";

    public string RunCommand { get; set; } = "pwsh /app/main.ps1";

    public Dictionary<string, string> OtherFiles { get; set; } = new();
}