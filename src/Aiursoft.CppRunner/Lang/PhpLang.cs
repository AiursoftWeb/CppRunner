namespace Aiursoft.CppRunner.Lang;

public class PhpLang : ILang
{
    public string LangDisplayName => "PHP (8.3.0)";

    public string LangExtension => "php";

    public string LangName => "php";

    public string DefaultCode =>
        """
        <?php
        function fibonacci() {
            $current = 1;
            $next = 1;
            while (true) {
                yield $current;
                $temp = $current;
                $current = $next;
                $next = $temp + $next;
            }
        }

        $generator = fibonacci();
        for ($i = 0; $i < 20; $i++) {
            echo $generator->current() . "\n";
            $generator->next();
        }
        """;

    public string EntryFileName => "main.php";

    public string DockerImage => "hub.aiursoft.cn/php:8.3.0-zts";

    public string RunCommand => "php /app/main.php";

    public Dictionary<string, string> OtherFiles => new();
    public bool NeedGpu => false;

}