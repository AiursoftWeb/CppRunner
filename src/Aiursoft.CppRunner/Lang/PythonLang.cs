namespace Aiursoft.CppRunner.Lang;

public class PythonLang : ILang
{
    public string LangDisplayName => "Python (CPython 3.11)";

    public string LangExtension => "python";

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

    public string LangExtension => "python";

    public string LangName => "python-pytorch";

    public string DefaultCode =>
        """
        import torch

        def matrix_power_iterative(A, n):
            """
            Computes A^n using an iterative approach (divide-and-conquer).
            Assumes A is a square matrix.
            """
            if n == 0:
                # Identity matrix
                return torch.eye(A.shape[0], dtype=A.dtype)

            result = torch.eye(A.shape[0], dtype=A.dtype)
            while n > 0:
                if n % 2 == 1:
                    result = torch.matmul(result, A)
                A = torch.matmul(A, A)
                n //= 2

            return result

        # Example usage
        A = torch.tensor([[1, 1], [1, 0]], dtype=torch.float32)  # Fibonacci matrix
        n = 10
        result = matrix_power_iterative(A, n - 1)
        print(result[0,0].item())
        """;

    public string EntryFileName => "main.py";

    public string DockerImage => "hub.aiursoft.cn/pytorch/pytorch:2.3.0-cuda11.8-cudnn8-devel";

    public string RunCommand => "python3 /app/main.py";
    public Dictionary<string, string> OtherFiles => new();
    public bool NeedGpu => true;
}