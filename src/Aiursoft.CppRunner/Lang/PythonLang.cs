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

public class PythonWithPytorch : ILang
{
    public string LangDisplayName => "Python with PyTorch (Pytorch 2.3.0; cuda 11.8; cudnn 8)";

    public string LangExtension => "py";

    public string LangName => "python-pytorch";

    public string DefaultCode =>
        """
        import torch

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if device.type == 'cuda':
            print("Using GPU")
            print("GPU Name:", torch.cuda.get_device_name(0))
            print("GPU Capability:", torch.cuda.get_device_capability(0))
        else:
            print("Using CPU")

        def fibonacci(n, device):
            fib_sequence = torch.zeros(n, dtype=torch.int64, device=device)
            fib_sequence[0] = 1
            fib_sequence[1] = 1
            
            for i in range(2, n):
                fib_sequence[i] = fib_sequence[i-1] + fib_sequence[i-2]
            
            return fib_sequence

        n = 20
        fib_sequence = fibonacci(n, device)
        print("Fibonacci sequence:", fib_sequence.cpu().numpy())
        """;

    public string EntryFileName => "main.py";

    public string DockerImage => "pytorch/pytorch:2.3.0-cuda11.8-cudnn8-devel";

    public string RunCommand => "python3 /app/main.py";
    public Dictionary<string, string> OtherFiles => new();
    public bool NeedGpu => true;
}