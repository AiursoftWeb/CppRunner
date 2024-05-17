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
            int idx = threadIdx.x;
            if (idx == 0) {
                fib[0] = 0;
            } else if (idx == 1) {
                fib[1] = 1;
            } else if (idx < n) {
                for (int i = 2; i <= idx; ++i) {
                    fib[i] = fib[i - 1] + fib[i - 2];
                }
            }
        }

        void computeFibonacci(long long *h_fib, int n) {
            long long *d_fib;
            cudaMalloc((void**)&d_fib, n * sizeof(long long));
            cudaMemcpy(d_fib, h_fib, n * sizeof(long long), cudaMemcpyHostToDevice);

            // Launch kernel with one block of n threads
            fibonacciKernel<<<1, n>>>(d_fib, n);
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