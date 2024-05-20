namespace Aiursoft.CppRunner.Lang;

public class TypeScriptLang : ILang
{
    public string LangDisplayName => "TypeScript (4.9.3, node 16.8.1)";

    public string LangExtension => "typescript";

    public string LangName => "typescript";

    public string DefaultCode =>
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

    public string EntryFileName => "main.ts";
    public string DockerImage => "hub.aiursoft.cn/vminnovations/typescript-sdk:16-latest";
    public string RunCommand => "tsc /app/main.ts && node /app/main.js";

    public Dictionary<string, string> OtherFiles => new();
    public bool NeedGpu => false;

}