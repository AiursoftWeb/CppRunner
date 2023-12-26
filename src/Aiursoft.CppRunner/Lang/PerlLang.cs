namespace Aiursoft.CppRunner.Lang;

public class PerlLang : ILang
{
    public string LangDisplayName => "Perl (5.39.5)";

    public string LangExtension => "pl";

    public string LangName => "perl";

    public string DefaultCode =>
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

    public string EntryFileName => "main.pl";

    public string DockerImage => "perl:5.39.5";

    public string RunCommand => "perl /app/main.pl";

    public Dictionary<string, string> OtherFiles => new();
}