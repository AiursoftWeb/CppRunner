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

        #define MAT_SIZE 2
        #define FIB_NUMS 20

        __global__ void matrixMultiply(int *a, int *b, int *c, int n) {
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

        void matrixMultiplyHost(int *a, int *b, int *c, int n) {
            int size = n * n * sizeof(int);

            int *d_a, *d_b, *d_c;

            // Allocate memory on the GPU
            cudaMalloc((void**)&d_a, size);
            cudaMalloc((void**)&d_b, size);
            cudaMalloc((void**)&d_c, size);

            // Copy matrices to the GPU
            cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
            cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

            // Define block and grid dimensions
            dim3 dimBlock(2, 2);
            dim3 dimGrid((n + dimBlock.x - 1) / dimBlock.x, (n + dimBlock.y - 1) / dimBlock.y);

            // Launch the kernel
            matrixMultiply<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, n);

            // Copy the result back to the CPU
            cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

            // Free GPU memory
            cudaFree(d_a);
            cudaFree(d_b);
            cudaFree(d_c);
        }

        void matrixPower(int *result, int power) {
            int base[MAT_SIZE * MAT_SIZE] = {1, 1, 1, 0};
            int temp[MAT_SIZE * MAT_SIZE];

            // Initialize result as identity matrix
            result[0] = 1;
            result[1] = 0;
            result[2] = 0;
            result[3] = 1;

            while (power) {
                if (power % 2 == 1) {
                    matrixMultiplyHost(result, base, temp, MAT_SIZE);
                    std::copy(temp, temp + MAT_SIZE * MAT_SIZE, result);
                }
                matrixMultiplyHost(base, base, temp, MAT_SIZE);
                std::copy(temp, temp + MAT_SIZE * MAT_SIZE, base);
                power /= 2;
            }
        }

        void computeFibonacci(int *fibs, int count) {
            int result[MAT_SIZE * MAT_SIZE];

            fibs[0] = 0; // F(0)
            fibs[1] = 1; // F(1)
            for (int i = 2; i < count; ++i) {
                matrixPower(result, i - 1);
                fibs[i] = result[0] + result[1]; // F(n) is the sum of the first row of the result matrix
            }
        }

        int main() {
            int fibs[FIB_NUMS];

            computeFibonacci(fibs, FIB_NUMS);

            std::cout << "First " << FIB_NUMS << " Fibonacci numbers:" << std::endl;
            for (int i = 0; i < FIB_NUMS; ++i) {
                std::cout << fibs[i] << " ";
            }
            std::cout << std::endl;

            return 0;
        }


        """;
    
    public string EntryFileName => "main.cu";
    public string DockerImage => "hub.aiursoft.cn/aiursoft/internalimages/nvidia";
    public string RunCommand => "nvcc /app/main.cu -o /app/main && /app/main";
    public bool NeedGpu => true;
    public Dictionary<string, string> OtherFiles => new();
}