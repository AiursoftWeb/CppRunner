namespace Aiursoft.CppRunner.Lang;

public class NodeJSLang : ILang
{
    public string LangDisplayName { get; set; } = "Javascript (Node.js v21)";
    
    public string LangExtension { get; set; } = "js";

    public string LangName { get; set; } = "javascript";

    public string DefaultCode { get; set; } = @"console.log('Hello, world!');";

    public string EntryFileName { get; set; } = "main.js";
    public string DockerImage { get; set; } = "node:21-alpine";
    public string RunCommand { get; set; } = "node /app/main.js";

    public Dictionary<string, string> OtherFiles { get; set; } = new();
}