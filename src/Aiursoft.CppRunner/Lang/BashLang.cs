namespace Aiursoft.CppRunner.Lang;

public class BashLang : ILang
{
    public string LangDisplayName { get; set; } = "Bash (on Ubuntu 24.04)";
    
    public string LangExtension { get; set; } = "sh";
    
    public string LangName { get; set; } = "bash";
    
    public string DefaultCode { get; set; } = 
        """
        #!/bin/bash
        a=1
        b=1
        count=1
        while [ $count -le 20 ]
        do
            echo $a
            temp=$a
            a=$b
            b=$((a + temp))
            count=$((count + 1))
        done
        """;
    
    public string EntryFileName { get; set; } = "main.sh";
    
    public string DockerImage { get; set; } = "ubuntu:24.04";
    
    public string RunCommand { get; set; } = "bash /app/main.sh";
    
    public Dictionary<string, string> OtherFiles { get; set; } = new();
}