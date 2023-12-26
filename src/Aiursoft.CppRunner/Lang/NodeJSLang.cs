namespace Aiursoft.CppRunner.Lang;

public class NodeJsLang : ILang
{
    public string LangDisplayName => "Javascript (Node.js v21)";

    public string LangExtension => "js";

    public string LangName => "javascript";

    public string DefaultCode =>
        """
        function fibonacci() {
          let current = 1, next = 1;
          return function() {
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

    public string EntryFileName => "main.js";
    public string DockerImage => "node:21-alpine";
    public string RunCommand => "node /app/main.js";

    public Dictionary<string, string> OtherFiles => new();
}