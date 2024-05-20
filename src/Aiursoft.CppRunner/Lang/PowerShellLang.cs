namespace Aiursoft.CppRunner.Lang;

public class PowerShellLang : ILang
{
    public string LangDisplayName => "PowerShell Core (Ubuntu 22.04)";

    public string LangExtension => "powershell";

    public string LangName => "powershell";

    public string DefaultCode =>
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

    public string EntryFileName => "main.ps1";

    public string DockerImage => "hub.aiursoft.cn/mcr.microsoft.com/powershell:latest";

    public string RunCommand => "pwsh /app/main.ps1";

    public Dictionary<string, string> OtherFiles => new();
    public bool NeedGpu => false;

}
