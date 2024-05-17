namespace Aiursoft.CppRunner.Lang;

public class CudaLang : ILang
{
    public string LangDisplayName => "CUDA 11.6";
    public string LangName => "cuda";
    public string LangExtension => "cu";

    public string DefaultCode =>
        """
        #include <iostream>

        __global__ void helloFromGPU() {
            printf("Hello World from GPU!\n");
        }

        int main() {
            helloFromGPU<<<1, 1>>>();
            cudaDeviceSynchronize();
            return 0;
        }
        """;
    
    public string EntryFileName => "main.cu";
    public string DockerImage => "nvidia/cuda:11.6.2-base-ubuntu20.04";
    public string RunCommand => "nvcc /app/main.cu -o /app/main && /app/main";
    public bool NeedGpu => true;
    public Dictionary<string, string> OtherFiles => new();
}