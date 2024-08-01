namespace Aiursoft.CppRunner.Lang;

public class CLang : ILang
{
    public string LangDisplayName => "C (gcc 9.5.0)";

    public string LangExtension => "c";

    public string LangName => "c";

    public string DefaultCode =>
        """
        #include <stdio.h>

        typedef struct FibonacciGenerator {
            int current;
            int next;
        } FibonacciGenerator;

        FibonacciGenerator fibonacci() {
            FibonacciGenerator fg = {1, 1};
            return fg;
        }

        int next(FibonacciGenerator* fg) {
            int result = fg->current;
            fg->current = fg->next;
            fg->next = fg->current + result;
            return result;
        }

        int main() {
            FibonacciGenerator fg = fibonacci();
            for (int i = 0; i < 20; i++) {
                printf("%d\n", next(&fg));
            }
            return 0;
        }

        """;

    public string EntryFileName => "main.c";

    public string DockerImage => "gcc:9.5.0";

    public string RunCommand => "gcc -Wall -Wextra -O2 /app/main.c -o /tmp/main && /tmp/main";

    public Dictionary<string, string> OtherFiles => new();
    public bool NeedGpu => false;

}