namespace Aiursoft.CppRunner.Lang;

public class GoLang : ILang
{
    public string LangDisplayName => "Go (Golang 1.21.5)";

    public string LangExtension => "go";

    public string LangName => "go";

    public string DefaultCode =>
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

    public string EntryFileName => "main.go";

    public string DockerImage => "golang:1.21.5";

    public string RunCommand => "go run /app/main.go";

    public Dictionary<string, string> OtherFiles => new();
}