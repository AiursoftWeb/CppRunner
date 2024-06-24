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

        # Print CUDA device information if available
        if torch.cuda.is_available():
            print(f"CUDA is available. Device count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"Device {i}: {torch.cuda.get_device_name(i)} - Capability: {torch.cuda.get_device_capability(i)}")
            device = torch.device("cuda")
            print("Running on GPU.")
        else:
            print("CUDA is not available. Running on CPU.")
            device = torch.device("cpu")

        def matrix_power_iterative(A, n):
            if n == 0:
                # Identity matrix
                return torch.eye(A.shape[0], dtype=A.dtype, device=device)

            result = torch.eye(A.shape[0], dtype=A.dtype, device=device)
            while n > 0:
                if n % 2 == 1:
                    result = torch.matmul(result, A)
                A = torch.matmul(A, A)
                n //= 2

            return result

        # Example usage
        A = torch.tensor([[1, 1], [1, 0]], dtype=torch.float32, device=device)  # Fibonacci matrix
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