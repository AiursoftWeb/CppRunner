namespace Aiursoft.CppRunner.Lang;

public class SwiftLang : ILang
{
    public string LangDisplayName { get; set; } = "Swift (Swift 5.5)";

    public string LangExtension { get; set; } = "swift";

    public string LangName { get; set; } = "swift";

    public string DefaultCode { get; set; } =
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

    public string EntryFileName { get; set; } = "main.swift";

    public string DockerImage { get; set; } = "swift:5.5-alpine";

    public string RunCommand { get; set; } = "swift /app/main.swift";

    public Dictionary<string, string> OtherFiles { get; set; } = new();
}