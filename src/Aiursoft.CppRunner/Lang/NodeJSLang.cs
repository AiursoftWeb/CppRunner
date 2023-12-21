namespace Aiursoft.CppRunner.Lang;

public class NodeJsLang : ILang
{
    public string LangDisplayName { get; set; } = "Javascript (Node.js v21)";
    
    public string LangExtension { get; set; } = "js";

    public string LangName { get; set; } = "javascript";

    public string DefaultCode { get; set; } = 
"""
function fibonacci() {
  let current = 0, next = 1;
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

    public string EntryFileName { get; set; } = "main.js";
    public string DockerImage { get; set; } = "node:21-alpine";
    public string RunCommand { get; set; } = "node /app/main.js";

    public Dictionary<string, string> OtherFiles { get; set; } = new();
}