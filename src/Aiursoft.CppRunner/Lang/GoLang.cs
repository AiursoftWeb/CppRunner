namespace Aiursoft.CppRunner.Lang;

public class GoLang : ILang
{
    public string LangDisplayName { get; set; } = "Go (Golang 1.21.5)";

    public string LangExtension { get; set; } = "go";

    public string LangName { get; set; } = "go";

    public string DefaultCode { get; set; } =
        """
        package main

        import "fmt"

        func fibonacci() func() int {
            current, next := 1, 1
            return func() int {
                result := current
                current, next = next, current + next
                return result
            }
        }

        func main() {
            fib := fibonacci()
            for i := 0; i < 20; i++ {
                fmt.Println(fib())
            }
        }
        """;

    public string EntryFileName { get; set; } = "main.go";

    public string DockerImage { get; set; } = "golang:1.21.5";

    public string RunCommand { get; set; } = "go run /app/main.go";

    public Dictionary<string, string> OtherFiles { get; set; } = new();
}