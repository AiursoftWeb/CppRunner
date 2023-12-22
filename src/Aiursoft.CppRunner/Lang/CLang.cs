namespace Aiursoft.CppRunner.Lang;

public class CLang : ILang
{
    public string LangDisplayName { get; set; } = "C (GNU GCC, stdc17)";
    
    public string LangExtension { get; set; } = "c";
    
    public string LangName { get; set; } = "c";

    public string DefaultCode { get; set; } =
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
    
    public string EntryFileName { get; set; } = "main.c";
    
    public string DockerImage { get; set; } = "frolvlad/alpine-gcc";
    
    public string RunCommand { get; set; } = "gcc -Wall -Wextra -O2 -std=c17 /app/main.c -o /tmp/main && /tmp/main";
    
    public Dictionary<string, string> OtherFiles { get; set; } = new();
}