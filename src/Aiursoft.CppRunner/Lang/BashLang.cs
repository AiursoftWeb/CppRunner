namespace Aiursoft.CppRunner.Lang;

public class BashLang : ILang
{
    public string LangDisplayName => "Bash (on Ubuntu 25.10)";

    public string LangExtension => "bash";

    public string LangName =>"bash";

    public string DefaultCode =>
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

    public string EntryFileName =>"main.sh";

    public string DockerImage =>"ubuntu:25.10";

    public string RunCommand =>"bash /app/main.sh";

    public Dictionary<string, string> OtherFiles =>new();
    public bool NeedGpu => false;
}
