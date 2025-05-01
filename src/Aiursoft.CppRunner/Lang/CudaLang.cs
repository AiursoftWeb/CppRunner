namespace Aiursoft.CppRunner.Lang;

public class CudaLang : ILang
{
    //12.8.1-devel-ubuntu24.04
    public string LangDisplayName => "CUDA 12.6.2 (on Ubuntu 24.04)";
    public string LangName => "cuda";
    public string LangExtension => "cpp";

    public string DefaultCode =>
        """
        #include <iostream>
        #include <cuda_runtime.h>
        #include <stdio.h>
        #include <chrono>

        // CUDA kernel for calculating Pi using the Leibniz formula
        __global__ void calculate_pi_gpu(double* partial_sums, long long num_iterations) {
            int tid = blockIdx.x * blockDim.x + threadIdx.x;
            int stride = blockDim.x * gridDim.x;

            double sum = 0.0;

            for (long long k = tid; k < num_iterations; k += stride) {
                double term = (k % 2 == 0) ? 1.0 : -1.0;
                term /= (2.0 * k + 1.0);
                sum += term;
            }

            partial_sums[tid] = sum;
        }

        // CPU function for calculating Pi using the same Leibniz formula
        double calculate_pi_cpu(long long num_iterations) {
            double sum = 0.0;

            for (long long k = 0; k < num_iterations; k++) {
                double term = (k % 2 == 0) ? 1.0 : -1.0;
                term /= (2.0 * k + 1.0);
                sum += term;
            }

            return 4.0 * sum;
        }

        int main() {
            // Number of iterations determines precision
            long long num_iterations = 4000000000; // 4 billion iterations for good precision

            // Variables for timing
            std::chrono::high_resolution_clock::time_point start, end;
            double cpu_time, gpu_time;

            // ----- CPU Implementation -----
            printf("Starting CPU calculation with %lld iterations...\n", num_iterations);
            start = std::chrono::high_resolution_clock::now();

            double pi_cpu = calculate_pi_cpu(num_iterations);

            end = std::chrono::high_resolution_clock::now();
            cpu_time = std::chrono::duration<double>(end - start).count();
            printf("CPU calculation complete in %.4f seconds\n", cpu_time);

            // ----- GPU Implementation -----
            printf("Starting GPU calculation with %lld iterations...\n", num_iterations);
            start = std::chrono::high_resolution_clock::now();

            // CUDA configuration
            int threadsPerBlock = 256;
            int blocks = 512; // Adjust based on your GPU
            int total_threads = blocks * threadsPerBlock;

            // Allocate device memory
            double* d_partial_sums;
            cudaMalloc(&d_partial_sums, total_threads * sizeof(double));

            // Execute kernel
            calculate_pi_gpu<<<blocks, threadsPerBlock>>>(d_partial_sums, num_iterations);

            // Allocate and copy partial sums from device to host
            double* h_partial_sums = new double[total_threads];
            cudaMemcpy(h_partial_sums, d_partial_sums, total_threads * sizeof(double), cudaMemcpyDeviceToHost);

            // Combine partial sums on the host
            double pi_gpu = 0.0;
            for (int i = 0; i < total_threads; i++) {
                pi_gpu += h_partial_sums[i];
            }
            pi_gpu *= 4.0; // The Leibniz formula gives pi/4

            end = std::chrono::high_resolution_clock::now();
            gpu_time = std::chrono::duration<double>(end - start).count();
            printf("GPU calculation complete in %.4f seconds\n", gpu_time);

            // Print results
            printf("\nResults:\n");
            printf("Pi (CPU):           %.16f\n", pi_cpu);
            printf("Pi (GPU):           %.16f\n", pi_gpu);
            printf("Math library value: %.16f\n", M_PI);
            printf("\nPerformance:\n");
            printf("CPU time: %.4f seconds\n", cpu_time);
            printf("GPU time: %.4f seconds\n", gpu_time);
            printf("Speedup: %.2fx\n", cpu_time / gpu_time);

            // Free memory
            delete[] h_partial_sums;
            cudaFree(d_partial_sums);

            return 0;
        }
        """;

    public string EntryFileName => "main.cu";
    public string DockerImage => "hub.aiursoft.cn/aiursoft/internalimages/nvidia";
    public string RunCommand => "nvcc /app/main.cu -o /app/main && /app/main";
    public bool NeedGpu => true;
    public Dictionary<string, string> OtherFiles => new();
}
