// reduction_naive_atomic.cu
// Naive atomic vector reduction (sum) with clean benchmarking + correctness run
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cmath>
#include <algorithm>

#define CUDA_CHECK(call) do {                                              \
    cudaError_t _err = (call);                                              \
    if (_err != cudaSuccess) {                                              \
        fprintf(stderr, "CUDA error %s:%d: %s (%d)\n",                      \
                __FILE__, __LINE__, cudaGetErrorString(_err), (int)_err);   \
        std::exit(1);                                                      \
    }                                                                       \
} while (0)

static void init_vector(float* h, int N) {
    for (int i = 0; i < N; i++) {
        h[i] = (i % 56) * 0.75f + 0.063f;
    }
}

static float cpu_sum(const float* A, int N) {
    double acc = 0.0; // use double for stable reference
    for (int i = 0; i < N; i++) acc += (double)A[i];
    return (float)acc;
}

static float abs_err(float a, float b) {
    return fabsf(a - b);
}

__global__ void vec_red_add_atomic(const float* __restrict__ d_A,
                                   float* d_sum, int N)
{
    for (int i = blockDim.x * blockIdx.x + threadIdx.x;
         i < N;
         i += blockDim.x * gridDim.x)
    {
        atomicAdd(d_sum, d_A[i]);
    }
}

static float time_kernel_ms(const float* d_A, float* d_sum, int N,
                            int grid, int block, int warmup, int iters)
{
    // Warmup (do NOT care about final d_sum here)
    CUDA_CHECK(cudaMemset(d_sum, 0, sizeof(float)));
    for (int i = 0; i < warmup; i++) {
        vec_red_add_atomic<<<grid, block>>>(d_A, d_sum, N);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed region (still do NOT care about correctness of accumulated result)
    CUDA_CHECK(cudaMemset(d_sum, 0, sizeof(float)));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; i++) {
        vec_red_add_atomic<<<grid, block>>>(d_A, d_sum, N);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaGetLastError());

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return total_ms / iters;
}

int main(int argc, char** argv)
{
    // Args: N iters block
    int N     = (argc > 1) ? std::atoi(argv[1]) : 1000000;
    int iters = (argc > 2) ? std::atoi(argv[2]) : 200;
    int block = (argc > 3) ? std::atoi(argv[3]) : 256;

    if (N <= 0) N = 4096;
    if (iters <= 0) iters = 200;
    if (block <= 0) block = 256;

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s\n", prop.name);
    printf("=== Naive Atomic Reduction ===\n");
    printf("N=%d, iters=%d, block=%d\n", N, iters, block);

    size_t bytes = (size_t)N * sizeof(float);

    float* h_A = (float*)std::malloc(bytes);
    if (!h_A) {
        fprintf(stderr, "Host malloc failed. Try smaller N.\n");
        return 1;
    }
    init_vector(h_A, N);

    float* d_A = nullptr;
    float* d_sum = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_sum, sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));

    // Grid choice: cap to avoid absurd contention; correctness unaffected (grid-stride loop).
    int grid = (N + block - 1) / block;
    int grid_cap = prop.multiProcessorCount * 20;
    grid = std::min(grid, grid_cap);

    printf("grid=%d (cap=%d)\n", grid, grid_cap);

    // --- Debug: verify memset works ---
    CUDA_CHECK(cudaMemset(d_sum, 0, sizeof(float)));
    float tmp = -1.0f;
    CUDA_CHECK(cudaMemcpy(&tmp, d_sum, sizeof(float), cudaMemcpyDeviceToHost));
    printf("Debug: after memset, d_sum=%f (expected 0.0)\n", tmp);

    // --- Benchmark (kernel time) ---
    int warmup = 10;
    float ms = time_kernel_ms(d_A, d_sum, N, grid, block, warmup, iters);

    // --- Correctness run (single pass, fresh zero) ---
    CUDA_CHECK(cudaMemset(d_sum, 0, sizeof(float)));
    vec_red_add_atomic<<<grid, block>>>(d_A, d_sum, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    float gpu_sum = 0.0f;
    CUDA_CHECK(cudaMemcpy(&gpu_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost));

    float ref = cpu_sum(h_A, N);
    float err = abs_err(gpu_sum, ref);

    // Reduction is contention-bound; bytes-read estimate is still useful as a sanity metric
    double seconds = (double)ms / 1e3;
    double bytes_read = 1.0 * (double)N * sizeof(float);
    double approx_read_gbps = (bytes_read / seconds) / 1e9;

    printf("naive atomic: avg=%.6f ms, approx_read=%.2f GB/s\n", ms, approx_read_gbps);
    printf("sum: gpu=%f  ref=%f  abs_err=%g\n", gpu_sum, ref, err);

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_sum));
    std::free(h_A);

    return 0;
}
