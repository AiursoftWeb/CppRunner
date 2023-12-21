namespace Aiursoft.CppRunner.Lang;

public class PerlLang : ILang
{
    public string LangDisplayName { get; set; } = "Perl (Perl 5.34)";

    public string LangExtension { get; set; } = "pl";

    public string LangName { get; set; } = "perl";

    public string DefaultCode { get; set; } =
        """
        sub fibonacci {
            my ($current, $next) = (1, 1);
            return sub {
                my $result = $current;
                ($current, $next) = ($next, $current + $next);
                return $result;
            };
        }

        my $fib = fibonacci();
        for (1..20) {
            print $fib->(), "\n";
        }
        """;

    public string EntryFileName { get; set; } = "main.pl";

    public string DockerImage { get; set; } = "perl:5.34-alpine";

    public string RunCommand { get; set; } = "perl /app/main.pl";

    public Dictionary<string, string> OtherFiles { get; set; } = new();
}