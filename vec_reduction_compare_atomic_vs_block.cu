// reduction_compare_atomic_vs_block.cu
// Compare naive atomic reduction vs block-level reduction (serial-in-thread0)
// Build: nvcc -O3 -lineinfo vec_reduction_compare_atomic_vs_block.cu -o reduce_compare
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
    double acc = 0.0;
    for (int i = 0; i < N; i++) acc += (double)A[i];
    return (float)acc;
}

static float abs_err(float a, float b) { return fabsf(a - b); }

// --- Naive atomic baseline ---
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

// --- Block-level: per-thread local sum + shared + serial block sum + 1 atomic per block ---
__global__ void vec_red_add_block_serial(const float* __restrict__ d_A,
                                         float* d_sum, int N)
{
    extern __shared__ float block_shmem[];

    float local = 0.0f;
    for (int idx = blockDim.x * blockIdx.x + threadIdx.x;
         idx < N;
         idx += blockDim.x * gridDim.x)
    {
        local += d_A[idx];
    }

    block_shmem[threadIdx.x] = local;
    __syncthreads();

    if (threadIdx.x == 0) {
        float block_sum = 0.0f;
        for (int i = 0; i < blockDim.x; i++) {
            block_sum += block_shmem[i];
        }
        atomicAdd(d_sum, block_sum);
    }
}

// Generic timing helper for the atomic kernel
static float time_atomic_ms(const float* d_A, float* d_sum, int N,
                            int grid, int block, int warmup, int iters)
{
    CUDA_CHECK(cudaMemset(d_sum, 0, sizeof(float)));
    for (int i = 0; i < warmup; i++) vec_red_add_atomic<<<grid, block>>>(d_A, d_sum, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemset(d_sum, 0, sizeof(float)));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; i++) vec_red_add_atomic<<<grid, block>>>(d_A, d_sum, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaGetLastError());

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return total_ms / iters;
}

// Generic timing helper for the block-serial kernel (NOTE the 3rd launch parameter for shmem)
static float time_block_serial_ms(const float* d_A, float* d_sum, int N,
                                  int grid, int block, int warmup, int iters)
{
    size_t shmem_bytes = (size_t)block * sizeof(float); // <<< IMPORTANT

    CUDA_CHECK(cudaMemset(d_sum, 0, sizeof(float)));
    for (int i = 0; i < warmup; i++)
        vec_red_add_block_serial<<<grid, block, shmem_bytes>>>(d_A, d_sum, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemset(d_sum, 0, sizeof(float)));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; i++)
        vec_red_add_block_serial<<<grid, block, shmem_bytes>>>(d_A, d_sum, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaGetLastError());

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return total_ms / iters;
}

static float correctness_run_atomic(const float* d_A, float* d_sum, int N, int grid, int block)
{
    CUDA_CHECK(cudaMemset(d_sum, 0, sizeof(float)));
    vec_red_add_atomic<<<grid, block>>>(d_A, d_sum, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    float out = 0.0f;
    CUDA_CHECK(cudaMemcpy(&out, d_sum, sizeof(float), cudaMemcpyDeviceToHost));
    return out;
}

static float correctness_run_block_serial(const float* d_A, float* d_sum, int N, int grid, int block)
{
    size_t shmem_bytes = (size_t)block * sizeof(float); // <<< IMPORTANT

    CUDA_CHECK(cudaMemset(d_sum, 0, sizeof(float)));
    vec_red_add_block_serial<<<grid, block, shmem_bytes>>>(d_A, d_sum, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    float out = 0.0f;
    CUDA_CHECK(cudaMemcpy(&out, d_sum, sizeof(float), cudaMemcpyDeviceToHost));
    return out;
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

    // Grid choice: cap (helps reduction kernels; correctness unaffected due to grid-stride)
    int grid = (N + block - 1) / block;
    int grid_cap = prop.multiProcessorCount * 20;
    grid = std::min(grid, grid_cap);
    printf("grid=%d (cap=%d)\n", grid, grid_cap);

    // Debug: verify memset works
    CUDA_CHECK(cudaMemset(d_sum, 0, sizeof(float)));
    float tmp = -1.0f;
    CUDA_CHECK(cudaMemcpy(&tmp, d_sum, sizeof(float), cudaMemcpyDeviceToHost));
    printf("Debug: after memset, d_sum=%f (expected 0.0)\n", tmp);

    int warmup = 10;

    // --- Benchmark ---
    float ms_atomic = time_atomic_ms(d_A, d_sum, N, grid, block, warmup, iters);
    float ms_block  = time_block_serial_ms(d_A, d_sum, N, grid, block, warmup, iters);

    // --- Correctness (single-pass runs) ---
    float gpu_atomic = correctness_run_atomic(d_A, d_sum, N, grid, block);
    float gpu_block  = correctness_run_block_serial(d_A, d_sum, N, grid, block);
    float ref        = cpu_sum(h_A, N);

    // Approx bytes read per run (not a great metric for reduction, but OK as a sanity number)
    auto approx_read_gbps = [&](float ms) {
        double seconds = (double)ms / 1e3;
        double bytes_read = 1.0 * (double)N * sizeof(float);
        return (bytes_read / seconds) / 1e9;
    };

    printf("\n--- Results (avg over %d iters) ---\n", iters);
    printf("naive atomic:      avg=%.6f ms  approx_read=%.2f GB/s  sum=%f  abs_err=%g\n",
           ms_atomic, approx_read_gbps(ms_atomic), gpu_atomic, abs_err(gpu_atomic, ref));
    printf("block serial/blk:  avg=%.6f ms  approx_read=%.2f GB/s  sum=%f  abs_err=%g\n",
           ms_block, approx_read_gbps(ms_block),  gpu_block,  abs_err(gpu_block,  ref));

    printf("\nSpeedup (atomic / block-serial) = %.2fx\n", ms_atomic / ms_block);

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_sum));
    std::free(h_A);
    return 0;
}
