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

        // 定义矩阵的大小（N x N）
        #define N 2

        __global__ void matrixMulKernel(int* a, int* b, int* c, int n) {
            int row = blockIdx.y * blockDim.y + threadIdx.y;
            int col = blockIdx.x * blockDim.x + threadIdx.x;
            int sum = 0;

            if (row < n && col < n) {
                for (int k = 0; k < n; ++k) {
                    sum += a[row * n + k] * b[k * n + col];
                }
                c[row * n + col] = sum;
            }
        }

        void matrixMul(int* a, int* b, int* c, int n) {
            int size = n * n * sizeof(int);
            int* d_a, * d_b, * d_c;

            // 分配设备内存
            cudaMalloc((void**)&d_a, size);
            cudaMalloc((void**)&d_b, size);
            cudaMalloc((void**)&d_c, size);

            // 复制数据到设备
            cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
            cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

            // 定义CUDA网格和块的大小
            dim3 threadsPerBlock(16, 16);
            dim3 blocksPerGrid((n + threadsPerBlock.x - 1) / threadsPerBlock.x, (n + threadsPerBlock.y - 1) / threadsPerBlock.y);

            // 调用CUDA内核
            matrixMulKernel <<<blocksPerGrid, threadsPerBlock >>> (d_a, d_b, d_c, n);

            // 复制结果回主机
            cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

            // 释放设备内存
            cudaFree(d_a);
            cudaFree(d_b);
            cudaFree(d_c);
        }

        int main() {
            int a[N * N], b[N * N], c[N * N];

            // | 1 2 |
            // | 3 4 |
            a[0] = 1;
            a[1] = 2;
            a[2] = 3;
            a[3] = 4;


            // | 5 6 |
            // | 7 8 |
            b[0] = 5;
            b[1] = 6;
            b[2] = 7;
            b[3] = 8;

            // 调用矩阵乘法函数
            matrixMul(a, b, c, N);

            // 输出结果矩阵c
            std::cout << "Result matrix:" << std::endl;
            for (int i = 0; i < N; ++i) {
                for (int j = 0; j < N; ++j) {
                    std::cout << c[i * N + j] << " ";
                }
                std::cout << std::endl;
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