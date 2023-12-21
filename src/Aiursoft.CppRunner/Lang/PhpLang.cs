namespace Aiursoft.CppRunner.Lang;

public class PhpLang : ILang
{
    public string LangDisplayName { get; set; } = "PHP (PHP 8.0)";

    public string LangExtension { get; set; } = "php";

    public string LangName { get; set; } = "php";

    public string DefaultCode { get; set; } =
        """
        <?php
        function fibonacci() {
            $current = 1;
            $next = 1;
            while (true) {
                yield $current;
                $next = $current + ($current = $next);
            }
        }

        $generator = fibonacci();
        for ($i = 0; $i < 20; $i++) {
            echo $generator->current() . "\n";
            $generator->next();
        }
        """;

    public string EntryFileName { get; set; } = "main.php";

    public string DockerImage { get; set; } = "php:8.0-alpine";

    public string RunCommand { get; set; } = "php /app/main.php";

    public Dictionary<string, string> OtherFiles { get; set; } = new();
}