namespace Aiursoft.CppRunner.Lang;

public class SwiftLang : ILang
{
    public string LangDisplayName => "Swift (5.8.1)";

    public string LangExtension => "swift";

    public string LangName => "swift";

    public string DefaultCode =>
        """
        func fibonacci() -> () -> Int {
            var current = 1, next = 1
            return {
                let result = current
                current = next
                next = current + result
                return result
            }
        }

        let fib = fibonacci()
        for _ in 0..<20 {
            print(fib())
        }
        """;

    public string EntryFileName => "main.swift";

    public string DockerImage => "swift:5.8.1";

    public string RunCommand => "swift /app/main.swift";

    public Dictionary<string, string> OtherFiles => new();
}