namespace Aiursoft.CppRunner.Lang;

public class TypeScriptLang : ILang
{
    public string LangDisplayName { get; set; } = "TypeScript (Node.js v21)";

    public string LangExtension { get; set; } = "ts";

    public string LangName { get; set; } = "typescript";

    public string DefaultCode { get; set; } =
        """
        function fibonacci(): () => number {
          let current = 1, next = 1;
          return function(): number {
              const temp = current;
              current = next;
              next = temp + current;
              return temp;
          };
        }

        const fib = fibonacci();
        for (let i = 0; i < 20; i++) {
          console.log(fib());
        }
        """;

    public string EntryFileName { get; set; } = "main.ts";
    public string DockerImage { get; set; } = "node:21-alpine";
    public string RunCommand { get; set; } = "tsc /app/main.ts && node /app/main.js";

    public Dictionary<string, string> OtherFiles { get; set; } = new();
}