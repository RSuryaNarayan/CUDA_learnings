// transpose_matrix_naive.cu
// Build:
//   nvcc -O3 transpose_matrix_naive.cu -o transpose
//
// Run:
//   ./transpose            (default N=4096, iters=200)
//   ./transpose 8192 100   (N=8192, iters=100)

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cmath>

#define CUDA_CHECK(call) do {                                              \
    cudaError_t _err = (call);                                              \
    if (_err != cudaSuccess) {                                              \
        fprintf(stderr, "CUDA error %s:%d: %s (%d)\n",                      \
                __FILE__, __LINE__, cudaGetErrorString(_err), (int)_err);   \
        std::exit(1);                                                      \
    }                                                                       \
} while (0)

static void init_matrix(float* h, int N) {
    for (int r = 0; r < N; r++) {
        for (int c = 0; c < N; c++) {
            h[r * N + c] = (float)((r * 131 + c * 17) % 1000) * 0.001f;
        }
    }
}

static float max_abs_err_transpose(const float* A, const float* B, int N) {
    // B should be transpose of A: B[c*N + r] == A[r*N + c]
    float m = 0.0f;
    for (int r = 0; r < N; r++) {
        for (int c = 0; c < N; c++) {
            float ref = A[r * N + c];
            float got = B[c * N + r];
            m = fmaxf(m, fabsf(ref - got));
        }
    }
    return m;
}

__global__ void transpose_naive(const float* __restrict__ in,
                                float* __restrict__ out,
                                int N)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < N && col < N) {
        out[col * N + row] = in[row * N + col];
    }
}

static float time_naive_ms(const float* dIn, float* dOut, int N,
                           dim3 grid, dim3 block,
                           int warmup, int iters)
{
    // Warmup (also catches errors early)
    for (int i = 0; i < warmup; i++) {
        transpose_naive<<<grid, block>>>(dIn, dOut, N);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; i++) {
        transpose_naive<<<grid, block>>>(dIn, dOut, N);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaGetLastError());

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return ms / iters;
}

int main(int argc, char** argv)
{
    int N = (argc > 1) ? std::atoi(argv[1]) : 4096;
    int iters = (argc > 2) ? std::atoi(argv[2]) : 200;
    int warmup = 10;
    if (N <= 0) N = 4096;
    if (iters <= 0) iters = 200;

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s\n", prop.name);
    printf("=== Naive transpose benchmark N=%d, iters=%d ===\n", N, iters);

    size_t bytes = (size_t)N * (size_t)N * sizeof(float);

    float* hIn  = (float*)std::malloc(bytes);
    float* hOut = (float*)std::malloc(bytes);
    if (!hIn || !hOut) {
        fprintf(stderr, "Host malloc failed. Try smaller N.\n");
        return 1;
    }
    init_matrix(hIn, N);

    float *dIn=nullptr, *dOut=nullptr;
    CUDA_CHECK(cudaMalloc(&dIn, bytes));
    CUDA_CHECK(cudaMalloc(&dOut, bytes));
    CUDA_CHECK(cudaMemcpy(dIn, hIn, bytes, cudaMemcpyHostToDevice));

    constexpr int TILE = 32;  // try 16 or 32
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE,
              (N + TILE - 1) / TILE);

    float ms = time_naive_ms(dIn, dOut, N, grid, block, warmup, iters);

    CUDA_CHECK(cudaMemcpy(hOut, dOut, bytes, cudaMemcpyDeviceToHost));
    float err = max_abs_err_transpose(hIn, hOut, N);

    // Effective bandwidth: read + write = 2 * bytes
    double seconds = (double)ms / 1e3;
    double moved = 2.0 * (double)bytes;
    double gbps = (moved / seconds) / 1e9;

    printf("block=(%d,%d) grid=(%d,%d)\n", block.x, block.y, grid.x, grid.y);
    printf("naive: avg=%.6f ms  approx=%.2f GB/s  max_err=%g\n", ms, gbps, err);

    CUDA_CHECK(cudaFree(dIn));
    CUDA_CHECK(cudaFree(dOut));
    std::free(hIn);
    std::free(hOut);
    return 0;
}
