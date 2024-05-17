namespace Aiursoft.CppRunner.Lang;

public class PythonLang : ILang
{
    public string LangDisplayName => "Python (CPython 3.11)";

    public string LangExtension => "py";

    public string LangName => "python";

    public string DefaultCode =>
        """
        def fibonacci():
            current, next = 1, 1
            while True:
                yield current
                current, next = next, current + next

        fib = fibonacci()
        for _ in range(20):
            print(next(fib))

        """;

    public string EntryFileName => "main.py";
    public string DockerImage => "hub.aiursoft.cn/python:3.11";
    public string RunCommand => "python3 /app/main.py";

    public Dictionary<string, string> OtherFiles => new();
    public bool NeedGpu => false;

}