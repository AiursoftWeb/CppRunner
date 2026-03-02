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
        #include <omp.h> // 引入 OpenMP 用于 CPU 多线程

        // 优化的 CUDA Kernel: 共享内存规约 + 原子加法，消除取模和分支
        __global__ void calculate_pi_gpu(double* d_pi, long long num_iterations) {
            // 动态分配共享内存
            extern __shared__ double sdata[];
            
            int tid = threadIdx.x;
            int global_id = blockIdx.x * blockDim.x + threadIdx.x;
            int stride = blockDim.x * gridDim.x;

            // 1. 网格步长循环计算局部和 (使用位运算 & 替代取模 %)
            double sum = 0.0;
            for (long long k = global_id; k < num_iterations; k += stride) {
                double sign = (k & 1) ? -1.0 : 1.0; 
                sum += sign / (2.0 * k + 1.0);
            }
            // 将线程的局部累加结果放入共享内存
            sdata[tid] = sum;
            __syncthreads(); // 确保 Block 内所有线程都已写入共享内存

            // 2. 共享内存并行规约 (Block 内部求和，大幅减少写入全局内存的次数)
            for (int s = blockDim.x / 2; s > 0; s >>= 1) {
                if (tid < s) {
                    sdata[tid] += sdata[tid + s];
                }
                __syncthreads();
            }

            // 3. 仅由每个 Block 的 0 号线程，将该 Block 的总和通过原子操作累加到全局内存
            if (tid == 0) {
                atomicAdd(d_pi, sdata[0]);
            }
        }

        // 优化的 CPU 函数: 启用多核多线程
        double calculate_pi_cpu(long long num_iterations) {
            double sum = 0.0;
            
            // OpenMP 编译制导指令：自动将循环分配给所有 CPU 核心，并安全地规约汇总 sum
            #pragma omp parallel for reduction(+:sum)
            for (long long k = 0; k < num_iterations; k++) {
                double sign = (k & 1) ? -1.0 : 1.0;
                sum += sign / (2.0 * k + 1.0);
            }
            return 4.0 * sum;
        }

        int main() {
            long long num_iterations = 4000000000;
            std::chrono::high_resolution_clock::time_point start, end;
            double cpu_time, gpu_time;

            // ----- 公平的 CPU 多核实现 -----
            printf("Starting Multi-core CPU calculation with %lld iterations...\n", num_iterations);
            start = std::chrono::high_resolution_clock::now();
            double pi_cpu = calculate_pi_cpu(num_iterations);
            end = std::chrono::high_resolution_clock::now();
            cpu_time = std::chrono::duration<double>(end - start).count();
            printf("CPU calculation complete in %.4f seconds\n", cpu_time);

            // ----- 专业的 GPU 实现 -----
            printf("Starting Optimized GPU calculation with %lld iterations...\n", num_iterations);
            start = std::chrono::high_resolution_clock::now();

            int threadsPerBlock = 256;
            int blocks = 512;

            // 只需要在显存中分配一个 double 大小的空间
            double* d_pi;
            cudaMalloc(&d_pi, sizeof(double));
            cudaMemset(d_pi, 0, sizeof(double)); // 初始化为 0

            // 启动 Kernel，注意第三个参数：动态分配给每个 Block 的共享内存大小
            calculate_pi_gpu<<<blocks, threadsPerBlock, threadsPerBlock * sizeof(double)>>>(d_pi, num_iterations);

            // 从显存取回最终的唯一结果
            double h_pi = 0.0;
            cudaMemcpy(&h_pi, d_pi, sizeof(double), cudaMemcpyDeviceToHost);
            double pi_gpu = h_pi * 4.0;

            end = std::chrono::high_resolution_clock::now();
            gpu_time = std::chrono::duration<double>(end - start).count();
            printf("GPU calculation complete in %.4f seconds\n", gpu_time);

            // Print results
            printf("\nResults:\n");
            printf("Pi (CPU):           %.16f\n", pi_cpu);
            printf("Pi (GPU):           %.16f\n", pi_gpu);
            printf("Math library value: %.16f\n", M_PI);

            printf("\nPerformance:\n");
            printf("CPU Multi-core time: %.4f seconds\n", cpu_time);
            printf("GPU Optimized time:  %.4f seconds\n", gpu_time);
            printf("True Speedup:        %.2fx\n", cpu_time / gpu_time);

            cudaFree(d_pi);
            return 0;
        }
        """;

    public string EntryFileName => "main.cu";
    public string DockerImage => "hub.aiursoft.com/aiursoft/internalimages/nvidia";
    public string RunCommand => "nvcc -O3 -arch=native -Xcompiler -fopenmp /app/main.cu -o pi_optimized && ./pi_optimized";
    public bool NeedGpu => true;
    public Dictionary<string, string> OtherFiles => new();
}
