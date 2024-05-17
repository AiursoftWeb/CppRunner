namespace Aiursoft.CppRunner.Lang;

public class CudaLang : ILang
{
    public string LangDisplayName => "CUDA 11.6 (on Ubuntu 20.04)";
    public string LangName => "cuda";
    public string LangExtension => "cpp";

    public string DefaultCode =>
        """
#include <iostream>
#include <cuda_runtime.h>

__global__ void helloFromGPU() {
    printf("Hello World from GPU!\n");
}

int main() {
    helloFromGPU<<<1, 1>>>();
    cudaError_t err = cudaGetLastError(); // Check for any errors launching the kernel
    if (err != cudaSuccess) {
        std::cerr << "Error: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }
    err = cudaDeviceSynchronize(); // Wait for the kernel to finish
    if (err != cudaSuccess) {
        std::cerr << "Error: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }
    return 0;
}

        """;
    
    public string EntryFileName => "main.cu";
    public string DockerImage => "hub.aiursoft.cn/aiursoft/internalimages/nvidia";
    public string RunCommand => "nvcc /app/main.cu -o /app/main && /app/main";
    public bool NeedGpu => true;
    public Dictionary<string, string> OtherFiles => new();
}