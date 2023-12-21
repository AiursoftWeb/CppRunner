namespace Aiursoft.CppRunner.Lang;

public class PythonLang : ILang
{
    public string LangDisplayName { get; set; } = "Python (CPython 3.11)";
    
    public string LangExtension { get; set; } = "py";

    public string LangName { get; set; } = "python";

    public string DefaultCode { get; set; } = @"print(""ciallo"")
def generate_fibonacci_sequence(n):
    fibonacci_sequence = []
    a, b = 0, 1
    for _ in range(n):
        fibonacci_sequence.append(a)
        a, b = b, a + b
    return fibonacci_sequence

print(generate_fibonacci_sequence(10))
    ";

    public string EntryFileName { get; set; } = "main.py";
    public string DockerImage { get; set; } = "docker.io/python:3.11-alpine";
    public string RunCommand { get; set; } = "python3 /app/main.py";

    public Dictionary<string, string> OtherFiles { get; set; } = new();
}