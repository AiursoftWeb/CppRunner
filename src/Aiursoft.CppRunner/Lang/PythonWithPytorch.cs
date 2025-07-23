namespace Aiursoft.CppRunner.Lang;

public class PythonWithPytorch : ILang
{
    //2.7.0-cuda12.6-cudnn9-devel
    public string LangDisplayName => "Python with PyTorch (Pytorch 2.7.0; cuda 12.6; cudnn 9)";

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

        def matrix_power_iterative(A, n):
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
        n = 100
        result = matrix_power_iterative(A, n - 1)
        print(result[0,0].item())
        """;

    public string EntryFileName => "main.py";

    public string DockerImage => "pytorch/pytorch:2.7.0-cuda12.6-cudnn9-devel";

    public string RunCommand => "python3 /app/main.py";
    public Dictionary<string, string> OtherFiles => new();
    public bool NeedGpu => true;
}
