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

        __global__ void fibonacciKernel(long long *fib, int n) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= n) return;

            long long a = 0, b = 1;
            for (int i = 2; i <= idx; ++i) {
                long long next = a + b;
                a = b;
                b = next;
            }
            fib[idx] = (idx == 0) ? 0 : (idx == 1) ? 1 : b;
        }

        void computeFibonacci(long long *h_fib, int n) {
            long long *d_fib;
            cudaMalloc((void**)&d_fib, n * sizeof(long long));

            int threadsPerBlock = 256;
            int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
            fibonacciKernel<<<blocksPerGrid, threadsPerBlock>>>(d_fib, n);
            cudaDeviceSynchronize();

            cudaMemcpy(h_fib, d_fib, n * sizeof(long long), cudaMemcpyDeviceToHost);
            cudaFree(d_fib);
        }

        int main() {
            const int n = 20;
            long long h_fib[n] = {0, 1}; // Initialize the first two values of the sequence
            computeFibonacci(h_fib, n);
            for (int i = 0; i < n; i++) {
                std::cout << h_fib[i] << std::endl;
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