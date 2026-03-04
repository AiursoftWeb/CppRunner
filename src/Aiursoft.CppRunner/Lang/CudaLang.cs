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
        #include <vector>
        #include <chrono>
        #include <omp.h>
        #include <cuda_runtime.h>

        // Macro for standard CUDA API error checking.
        #define CUDA_CHECK(call) \
            do { \
                cudaError_t err = call; \
                if (err != cudaSuccess) { \
                    std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
                    exit(EXIT_FAILURE); \
                } \
            } while(0)

        // ASCII palette mapping illumination intensity from low (dark) to high (bright).
        __device__ __host__ const char palette[] = " .,-~:;=!*#$@";

        // Integer square root implementation using Newton's method.
        // This ensures deterministic behavior across different hardware architectures 
        // by entirely eliminating floating-point rounding errors.
        __host__ __device__ int integer_sqrt(long long n) {
            long long x = n;
            long long y = (x + 1) / 2;
            while (y < x) {
                x = y;
                y = (x + n / x) / 2;
            }
            return (int)x;
        }

        // CUDA Kernel: Performs integer-based raycasting on a 2D grid.
        __global__ void render_sphere_gpu(char* d_out, int width, int height) {
            // Map thread execution to physical pixel coordinates.
            int x = blockIdx.x * blockDim.x + threadIdx.x;
            int y = blockIdx.y * blockDim.y + threadIdx.y;
            
            // Boundary check to prevent memory access violations.
            if (x >= width || y >= height) return;

            // Translate coordinates to place the origin (0,0) at the center of the canvas.
            long long vx = x - width / 2;
            long long vy = y - height / 2;
            long long R = 4000; // Sphere radius

            // Calculate depth using the sphere equation: Z^2 = R^2 - X^2 - Y^2
            long long z_sq = R * R - vx * vx - vy * vy;
            char pixel = ' '; // Default background pixel

            if (z_sq >= 0) { 
                // A non-negative Z^2 indicates an intersection with the sphere geometry.
                long long vz = integer_sqrt(z_sq);
                
                // Define a directional light vector (Lx, Ly, Lz).
                long long lx = -50, ly = -50, lz = 50;
                
                // Compute the dot product between the surface normal vector and the light vector.
                // This yields the diffuse illumination intensity.
                long long dot = vx * lx + vy * ly + vz * lz;
                if (dot < 0) dot = 0; // Clamp negative illumination (back-facing polygons) to 0.
                
                // Normalize the dot product to map it to the 13-character palette.
                // The theoretical maximum dot product is approximately 346410.
                int idx = (dot * 12) / 346410;
                if (idx > 12) idx = 12;
                
                pixel = palette[idx];
            }
            
            // Write the computed pixel to global device memory.
            d_out[y * width + x] = pixel;
        }

        // CPU Implementation: Functionally identical to the GPU kernel, utilizing OpenMP for parallelization.
        void render_sphere_cpu(char* h_out, int width, int height) {
            #pragma omp parallel for collapse(2)
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    long long vx = x - width / 2;
                    long long vy = y - height / 2;
                    long long R = 4000;
                    long long z_sq = R * R - vx * vx - vy * vy;
                    char pixel = ' ';
                    if (z_sq >= 0) {
                        long long vz = integer_sqrt(z_sq);
                        long long lx = -50, ly = -50, lz = 50;
                        long long dot = vx * lx + vy * ly + vz * lz;
                        if (dot < 0) dot = 0;
                        int idx = (dot * 12) / 346410;
                        if (idx > 12) idx = 12;
                        pixel = palette[idx];
                    }
                    h_out[y * width + x] = pixel;
                }
            }
        }

        // Helper function to render a downsampled version of the high-resolution buffer to the standard output.
        void print_ascii_canvas(const std::vector<char>& canvas, int width, const std::string& title) {
            std::cout << "\n========================================" << std::endl;
            std::cout << title << std::endl;
            std::cout << "========================================\n" << std::endl;
            
            // Downsampling logic: The step size in the Y-axis is twice that of the X-axis 
            // to compensate for the typical 2:1 aspect ratio of terminal fonts, preserving the spherical shape.
            for (int y = 1000; y < 9000; y += 200) {
                for (int x = 1000; x < 9000; x += 100) {
                    std::cout << canvas[y * width + x];
                }
                std::cout << '\n';
            }
        }

        int main() {
            // Define a 10000 x 10000 high-resolution canvas (100 million pixels) to ensure
            // a computationally intensive workload suitable for benchmarking.
            int width = 10000;
            int height = 10000;
            size_t size = width * height * sizeof(char);

            std::vector<char> h_out_cpu(width * height);
            std::vector<char> h_out_gpu(width * height);
            char* d_out;

            // CUDA Context initialization (warm-up) and memory allocation.
            CUDA_CHECK(cudaFree(0)); 
            CUDA_CHECK(cudaMalloc(&d_out, size));

            // Define the execution configuration parameters (Grid and Block dimensions).
            dim3 threads(16, 16);
            dim3 blocks((width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y);

            std::cout << "Rendering 100 Million Pixels 3D Sphere in Memory..." << std::endl;

            // --- GPU Execution and Profiling ---
            auto start_gpu = std::chrono::high_resolution_clock::now();
            render_sphere_gpu<<<blocks, threads>>>(d_out, width, height);
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(h_out_gpu.data(), d_out, size, cudaMemcpyDeviceToHost));
            auto end_gpu = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> gpu_time = end_gpu - start_gpu;

            // --- CPU Execution and Profiling ---
            auto start_cpu = std::chrono::high_resolution_clock::now();
            render_sphere_cpu(h_out_cpu.data(), width, height);
            auto end_cpu = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> cpu_time = end_cpu - start_cpu;

            // --- Output Results ---
            print_ascii_canvas(h_out_cpu, width, "     CPU RENDER (OPENMP MULTI-CORE)     ");
            print_ascii_canvas(h_out_gpu, width, "        GPU RENDER (CUDA KERNEL)        ");

            // --- Performance Metrics ---
            std::cout << "\n========================================" << std::endl;
            std::cout << "          PERFORMANCE METRICS           " << std::endl;
            std::cout << "========================================" << std::endl;
            std::cout << "CPU Time : " << cpu_time.count() << " seconds" << std::endl;
            std::cout << "GPU Time : " << gpu_time.count() << " seconds" << std::endl;
            std::cout << "----------------------------------------" << std::endl;
            std::cout << "Speedup  : GPU is " << (cpu_time.count() / gpu_time.count()) << "x FASTER" << std::endl;
            std::cout << "========================================" << std::endl;

            // Resource deallocation.
            CUDA_CHECK(cudaFree(d_out));
            return 0;
        }
        """;

    public string EntryFileName => "main.cu";
    public string DockerImage => "hub.aiursoft.com/aiursoft/internalimages/nvidia";
    public string RunCommand => "nvcc -O3 -arch=native -Xcompiler -fopenmp /app/main.cu -o pi_optimized && ./pi_optimized";
    public bool NeedGpu => true;
    public Dictionary<string, string> OtherFiles => new();
}
