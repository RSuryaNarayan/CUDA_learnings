// two_pass_reduction.cu
// Implement a kernel that outputs block-wise reductions 
// That can later be reduced either using the same kernel 
// or on CPU
// Build:
//   nvcc -O3 -lineinfo two_pass_reduce.cu -o two_pass
//
// Run:
//   ./two_pass
//
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cmath>
#include <vector>
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

// --------------------------- Kernels ---------------------------

// (1) Naive atomic baseline: every thread contributes via atomicAdd to one global scalar.
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

// (2) Block-level reduction, but intra-block sum is serial in thread 0 (simple but not optimal).
__global__ void vec_red_add_block_serial(const float* __restrict__ d_A,
                                         float* d_sum, int N)
{
    extern __shared__ float sdata[];

    float local = 0.0f;
    for (int idx = blockDim.x * blockIdx.x + threadIdx.x;
         idx < N;
         idx += blockDim.x * gridDim.x)
    {
        local += d_A[idx];
    }

    sdata[threadIdx.x] = local;
    __syncthreads();

    if (threadIdx.x == 0) {
        float block_sum = 0.0f;
        for (int i = 0; i < blockDim.x; i++) {
            block_sum += sdata[i];
        }
        atomicAdd(d_sum, block_sum);
    }
}

// (3) Block-level reduction with parallel tree reduction in shared memory.
// Requires blockDim.x to be a power of two for the simplest form.
__global__ void vec_red_add_block_tree(const float* __restrict__ d_A,
                                       float* d_sum, int N)
{
    extern __shared__ float sdata[];

    // Per-thread grid-stride accumulation into a register.
    float local = 0.0f;
    for (int idx = blockDim.x * blockIdx.x + threadIdx.x;
         idx < N;
         idx += blockDim.x * gridDim.x)
    {
        local += d_A[idx];
    }

    int tid = threadIdx.x;
    sdata[tid] = local;
    __syncthreads();

    // Tree reduction: halve active threads each step.
    // For block sizes like 128/256/512 this is ideal.
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(d_sum, sdata[0]); // one atomic per block
    }
}

// (4) Two-pass partial reductions. This is just 1 pass kernel
__global__ void reduce_pass_partials_serial(const float* __restrict__ d_A,
                                         float* d_sum, // the major difference is 
                                                       // d_sum is now a vector of shape gridDim.x
                                                       // with the partial sums
                                         int N)
{
    // as usual we create a container for shared
    extern __shared__ float sdata[];

    // grid-stride loop to handle 1 thread 
    // working on more than 1 element
    float local = 0.0f;
    for (int idx = blockDim.x * blockIdx.x + threadIdx.x;
         idx < N;
         idx += blockDim.x * gridDim.x)
    {
        local += d_A[idx];
    }

    // assign block data to shared mem container
    sdata[threadIdx.x] = local;
    __syncthreads();

    if (threadIdx.x == 0) {
        float block_sum = 0.0f;
        for (int i = 0; i < blockDim.x; i++) {
            block_sum += sdata[i];
        }
        d_sum[blockIdx.x] = block_sum;
        // atomicAdd(d_sum, block_sum); <--this would have been there in the previous kernels
        // The whole point of the two-pass reduction is to do away with any atomics
        // so we will just write out block level sums and call it recursively or reduce later
    }
}

// --------------------------- Timing helpers ---------------------------

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

static float time_block_serial_ms(const float* d_A, float* d_sum, int N,
                                  int grid, int block, int warmup, int iters)
{
    size_t shmem_bytes = (size_t)block * sizeof(float);

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

static float time_block_tree_ms(const float* d_A, float* d_sum, int N,
                                int grid, int block, int warmup, int iters)
{
    size_t shmem_bytes = (size_t)block * sizeof(float);

    CUDA_CHECK(cudaMemset(d_sum, 0, sizeof(float)));
    for (int i = 0; i < warmup; i++)
        vec_red_add_block_tree<<<grid, block, shmem_bytes>>>(d_A, d_sum, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemset(d_sum, 0, sizeof(float)));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; i++)
        vec_red_add_block_tree<<<grid, block, shmem_bytes>>>(d_A, d_sum, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaGetLastError());

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return total_ms / iters;
}

// Correctness: single-pass runs (fresh zero each time)
static float run_atomic_once(const float* d_A, float* d_sum, int N, int grid, int block)
{
    CUDA_CHECK(cudaMemset(d_sum, 0, sizeof(float)));
    vec_red_add_atomic<<<grid, block>>>(d_A, d_sum, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    float out = 0.0f;
    CUDA_CHECK(cudaMemcpy(&out, d_sum, sizeof(float), cudaMemcpyDeviceToHost));
    return out;
}

static float run_block_serial_once(const float* d_A, float* d_sum, int N, int grid, int block)
{
    size_t shmem_bytes = (size_t)block * sizeof(float);
    CUDA_CHECK(cudaMemset(d_sum, 0, sizeof(float)));
    vec_red_add_block_serial<<<grid, block, shmem_bytes>>>(d_A, d_sum, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    float out = 0.0f;
    CUDA_CHECK(cudaMemcpy(&out, d_sum, sizeof(float), cudaMemcpyDeviceToHost));
    return out;
}

static float run_block_tree_once(const float* d_A, float* d_sum, int N, int grid, int block)
{
    size_t shmem_bytes = (size_t)block * sizeof(float);
    CUDA_CHECK(cudaMemset(d_sum, 0, sizeof(float)));
    vec_red_add_block_tree<<<grid, block, shmem_bytes>>>(d_A, d_sum, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    float out = 0.0f;
    CUDA_CHECK(cudaMemcpy(&out, d_sum, sizeof(float), cudaMemcpyDeviceToHost));
    return out;
}

static float reduce_until_one_gpu(const float* d_in, int N,
                                  int block, int grid_cap,
                                  float** d_tmpA_out = nullptr,
                                  float** d_tmpB_out = nullptr)
{
    if ((block & (block - 1)) != 0) {
        fprintf(stderr, "reduce_until_one_gpu: block=%d not power of two\n", block);
        std::exit(1);
    }
    if (N <= 0) {
        fprintf(stderr, "reduce_until_one_gpu: N must be > 0\n");
        std::exit(1);
    }

    // First pass grid (and max temporary size needed)
    int grid1_needed = (N + block - 1) / block;
    int grid1 = (grid_cap > 0) ? std::min(grid1_needed, grid_cap) : grid1_needed;

    // Allocate two ping-pong buffers large enough for the biggest partial array (grid1)
    float* d_bufA = nullptr;
    float* d_bufB = nullptr;
    CUDA_CHECK(cudaMalloc(&d_bufA, (size_t)grid1 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_bufB, (size_t)grid1 * sizeof(float)));

    // We’ll alternate reading from src and writing into dst.
    const float* src = d_in;
    float* dst = d_bufA;

    int curN = N;
    size_t shmem = (size_t)block * sizeof(float);

    // Repeat until only 1 value remains
    while (true) {
        int grid_needed = (curN + block - 1) / block;
        int grid = (grid_cap > 0) ? std::min(grid_needed, grid_cap) : grid_needed;

        // Launch: src[0..curN) -> dst[0..grid)
        reduce_pass_partials_serial<<<grid, block, shmem>>>(src, dst, curN);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // After this pass, we have 'grid' partial sums
        curN = grid;

        if (curN == 1) {
            // Final answer is dst[0]
            float out = 0.0f;
            CUDA_CHECK(cudaMemcpy(&out, dst, sizeof(float), cudaMemcpyDeviceToHost));

            // Optionally return the temporary buffers to caller (so caller can reuse/free later)
            if (d_tmpA_out) *d_tmpA_out = d_bufA; else CUDA_CHECK(cudaFree(d_bufA));
            if (d_tmpB_out) *d_tmpB_out = d_bufB; else CUDA_CHECK(cudaFree(d_bufB));
            return out;
        }

        // Swap ping-pong buffers for next pass
        // Next pass reads from the buffer we just wrote.
        src = dst;
        dst = (dst == d_bufA) ? d_bufB : d_bufA;
    }
}


// --------------------------- Main ---------------------------

int main(int argc, char** argv)
{
    // Args: N iters block
    int N     = (argc > 1) ? std::atoi(argv[1]) : 1000000;
    int iters = (argc > 2) ? std::atoi(argv[2]) : 200;
    int block = (argc > 3) ? std::atoi(argv[3]) : 256;

    if (N <= 0) N = 4096;
    if (iters <= 0) iters = 200;
    if (block <= 0) block = 256;

    // Tree reduction kernel (as written) assumes power-of-two block sizes.
    // If you pass something else, it will still run but the reduction is not correct.
    if ((block & (block - 1)) != 0) {
        fprintf(stderr, "Error: block=%d is not a power of two. Use 128/256/512/etc.\n", block);
        return 1;
    }

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    int sm = prop.multiProcessorCount; //how many SMs

    printf("GPU: %s\n", prop.name);
    printf("SMs (multiProcessorCount) = %d\n", sm);
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

    //a sweep on the grid_cap as multiples of SM count. 

    int grid_needed = (N + block - 1) / block;

    int caps[] = { 1, 2, 4, 8, 16, 20, 40 }; // multiples of SM count

    printf("grid_needed=%d\n", grid_needed);
    printf("\n--- Grid sweep (cap = SM * k, grid = min(grid_needed, cap)) ---\n");
    printf("k\tgrid\tatomic_ms\tserial_ms\ttree_ms\tspeedup(a/serial)\tspeedup(a/tree)\n");

    int warmup = 10; // (keep as you already have)
    for (int k : caps) {
        int cap = sm * k;
        int grid = std::min(grid_needed, cap);

        float ms_atomic = time_atomic_ms(d_A, d_sum, N, grid, block, warmup, iters);
        float ms_serial = time_block_serial_ms(d_A, d_sum, N, grid, block, warmup, iters);
        float ms_tree   = time_block_tree_ms(d_A, d_sum, N, grid, block, warmup, iters);

        printf("%d\t%d\t%.6f\t%.6f\t%.6f\t%.2f\t\t\t%.2f\n",
            k, grid, ms_atomic, ms_serial, ms_tree,
            ms_atomic / ms_serial, ms_atomic / ms_tree);
    }

    // Also try "no cap" (full grid_needed)
    {
        int grid = grid_needed;
        float ms_atomic = time_atomic_ms(d_A, d_sum, N, grid, block, warmup, iters);
        float ms_serial = time_block_serial_ms(d_A, d_sum, N, grid, block, warmup, iters);
        float ms_tree   = time_block_tree_ms(d_A, d_sum, N, grid, block, warmup, iters);

        printf("nocap\t%d\t%.6f\t%.6f\t%.6f\t%.2f\t\t\t%.2f\n",
            grid, ms_atomic, ms_serial, ms_tree,
            ms_atomic / ms_serial, ms_atomic / ms_tree);
    }


    // Debug: verify memset works
    CUDA_CHECK(cudaMemset(d_sum, 0, sizeof(float)));
    float tmp = -1.0f;
    CUDA_CHECK(cudaMemcpy(&tmp, d_sum, sizeof(float), cudaMemcpyDeviceToHost));
    printf("Debug: after memset, d_sum=%f (expected 0.0)\n", tmp);
    
    warmup = 10;
    // Grid choice: cap to avoid launching tons of blocks for reduction.
    // We can switch back to 20 times for the rest
    int grid = (N + block - 1) / block;
    int grid_cap = prop.multiProcessorCount * 20;
    grid = std::min(grid, grid_cap);
    printf("grid=%d (cap=%d)\n", grid, grid_cap);
    // --- Benchmark (avg ms) ---
    float ms_atomic = time_atomic_ms(d_A, d_sum, N, grid, block, warmup, iters);
    float ms_serial = time_block_serial_ms(d_A, d_sum, N, grid, block, warmup, iters);
    float ms_tree   = time_block_tree_ms(d_A, d_sum, N, grid, block, warmup, iters);

    // --- Correctness (single-pass runs) ---
    float gpu_atomic = run_atomic_once(d_A, d_sum, N, grid, block);
    float gpu_serial = run_block_serial_once(d_A, d_sum, N, grid, block);
    float gpu_tree   = run_block_tree_once(d_A, d_sum, N, grid, block);
    float ref        = cpu_sum(h_A, N);

    auto approx_read_gbps = [&](float ms) {
        double seconds = (double)ms / 1e3;
        double bytes_read = 1.0 * (double)N * sizeof(float);
        return (bytes_read / seconds) / 1e9;
    };

    printf("\n--- Results (avg over %d iters) ---\n", iters);
    printf("naive atomic:        avg=%.6f ms  approx_read=%.2f GB/s  sum=%f  abs_err=%g\n",
           ms_atomic, approx_read_gbps(ms_atomic), gpu_atomic, abs_err(gpu_atomic, ref));
    printf("block serial (t0):   avg=%.6f ms  approx_read=%.2f GB/s  sum=%f  abs_err=%g\n",
           ms_serial, approx_read_gbps(ms_serial), gpu_serial, abs_err(gpu_serial, ref));
    printf("block tree (shared): avg=%.6f ms  approx_read=%.2f GB/s  sum=%f  abs_err=%g\n",
           ms_tree,   approx_read_gbps(ms_tree),   gpu_tree,   abs_err(gpu_tree,   ref));

    printf("\nSpeedups:\n");
    printf("  atomic / block-serial = %.2fx\n", ms_atomic / ms_serial);
    printf("  atomic / block-tree   = %.2fx\n", ms_atomic / ms_tree);
    printf("  serial / tree         = %.2fx\n", ms_serial / ms_tree);

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_sum));
    std::free(h_A);

  // testing multi-pass reduction (pass1 + pass2+pass3 + pass4 coded manually on GPU)
    {
        int N_small = 10;
        float h_A[N_small] = {1.0f, 2.0f, 5.0f, 8.0f, 13.0f, 45.0f, 123.5f, -1.0f, 4.0f, 10.0f};


        // device buffers
        float *d_A = nullptr;
        CUDA_CHECK(cudaMalloc(&d_A, (size_t)N_small * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_A, h_A, (size_t)N_small * sizeof(float), cudaMemcpyHostToDevice));

        // reduce in 1 shot
        cudaDeviceProp prop{};
        CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
        int sm = prop.multiProcessorCount;

        int block = 256;
        int grid_cap = sm * 8;   // start with 4–8×SM based on your sweep

        float gpu_sum = reduce_until_one_gpu(d_A, N_small, block, grid_cap);
        printf("multi-pass GPU sum = %f\n", gpu_sum);


        // CPU reference
        double ref = 0.0;
        for (int i = 0; i < N_small; i++) ref += (double)h_A[i];
        printf("\nCPU ref sum = %2.5f\n", (float)ref);

        CUDA_CHECK(cudaFree(d_A));
    }


    return 0;
}
