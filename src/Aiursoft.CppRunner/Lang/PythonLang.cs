namespace Aiursoft.CppRunner.Lang;

public class PythonLang : ILang
{
    public string LangDisplayName { get; set; } = "Python (CPython 3.11)";
    
    public string LangExtension { get; set; } = "py";

    public string LangName { get; set; } = "python";

    public string DefaultCode { get; set; } = 
"""
def fibonacci():
    current, next = 0, 1
    while True:
        yield current
        current, next = next, current + next

fib = fibonacci()
for _ in range(20):
    print(next(fib))
""";

    public string EntryFileName { get; set; } = "main.py";
    public string DockerImage { get; set; } = "python:3.11-alpine";
    public string RunCommand { get; set; } = "python3 /app/main.py";

    public Dictionary<string, string> OtherFiles { get; set; } = new();
}