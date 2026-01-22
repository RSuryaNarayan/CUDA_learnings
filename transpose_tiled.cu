// transpose_tiled.cu
// partly written by me (non-dynamic TILE logic), modified by chatGPT for dynamic tile size. 
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

__global__ void transpose_naive(const float* __restrict__ in,
                                float* __restrict__ out,
                                int N)
{
    int Xt = blockIdx.x * blockDim.x + threadIdx.x; // global X
    int Yt = blockIdx.y * blockDim.y + threadIdx.y; // global Y

    if (Xt < N && Yt < N) {
        // out[col][row] = in[row][col]
        out[Xt * N + Yt] = in[Yt * N + Xt];
    }
}

__global__ void transpose_tiled(const float* __restrict__ in,
                                float* __restrict__ out,
                                int N, int TILE)
{
    /*
    Load a small block-sized tile of the in array first into the shared memory.
    Say:
               | [a00 a01] |
     Tile =    |           |
               | [a10 a11] | _[2,2]
    */

    // Dynamic shared memory so TILE can be runtime-configurable.
    // We use padding (TILE+1) to avoid bank conflicts during tile[tx][ty] access.
    extern __shared__ float tile1d[];
    int pitch = TILE + 1; // padding
    auto TILE_AT = [&](int r, int c) -> float& { return tile1d[r * pitch + c]; };

    //fetch elements of in corresponding to tile
    int X_in_tile = blockIdx.x * TILE + threadIdx.x;
    int Y_in_tile = blockIdx.y * TILE + threadIdx.y;

    // tile[ty][tx] = in[Y_tile][X_tile]; <== this is what we want but note "in" is straightend out
    if (X_in_tile < N && Y_in_tile < N)
    {
        TILE_AT(threadIdx.y, threadIdx.x) = in[Y_in_tile * N + X_in_tile];
    }

    /* synchronize threads so that we have all data from the block*/

     __syncthreads();

    /*
    Use the loaded shared memory info to now write out shared memory elements into the out array!
               | [a00 a01] |
     Tile =    |           |
               | [a10 a11] | _[2,2]

    Here you want to be able to write out a00 a01 a10 a11 through the local block-scoped threadIdx.
    (X,Y) straightens it out column wise but again this results in strided access patters with TILE
    however in SHARED memory (this is cheap).
    out [sth, sth] = tile[threadIdx.x][threadIdx.y]
    the sth, sth must be the exact transpose of what we read into shared

    The reason the sth sth is the transpose of what we read in is simple. When you read in TILE,
    you are moving along X. This means you have to move down Y while writing the TILE out. Simple!

    But then you'd notice only blockIdx flips x,y but the added threadIdx.x remains the same.
    This is to ensure you write it out in row major order.

    Big Note: out  ==> resides in GLOBAL memory. These writes CANNOT be strided.
              TILE ==> resides in SHARED memory. These writes CAN be strided with much less cost!
    */
    int out_X = blockIdx.y * TILE + threadIdx.x;
    int out_Y = blockIdx.x * TILE + threadIdx.y;

    // NOTE: must bounds-check output too (edge tiles)
    if (out_X < N && out_Y < N) {
        out[out_Y * N + out_X] = TILE_AT(threadIdx.x, threadIdx.y);
    }
}

static void init_matrix(float* A, int N)
{
    for (int r = 0; r < N; r++) {
        for (int c = 0; c < N; c++) {
            A[r * N + c] = (float)((r * 131 + c * 17) % 1000) * 0.001f;
        }
    }
}

static float max_abs_err_transpose(const float* A, const float* B, int N)
{
    // B should satisfy: B[c*N + r] == A[r*N + c]
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

static float time_kernel_ms_naive(const float* dIn, float* dOut,
                                  int N, dim3 grid, dim3 block,
                                  int warmup, int iters)
{
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

static float time_kernel_ms_tiled(const float* dIn, float* dOut,
                                  int N, int TILE, dim3 grid, dim3 block,
                                  int warmup, int iters)
{
    size_t shmem_bytes = (size_t)TILE * (size_t)(TILE + 1) * sizeof(float);

    for (int i = 0; i < warmup; i++) {
        transpose_tiled<<<grid, block, shmem_bytes>>>(dIn, dOut, N, TILE);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; i++) {
        transpose_tiled<<<grid, block, shmem_bytes>>>(dIn, dOut, N, TILE);
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
    int N    = (argc > 1) ? std::atoi(argv[1]) : 4096;
    int TILE = (argc > 2) ? std::atoi(argv[2]) : 32;
    int iters = (argc > 3) ? std::atoi(argv[3]) : 200;

    if (N <= 0) N = 4096;
    if (TILE <= 0) TILE = 32;
    if (iters <= 0) iters = 200;

    // Basic constraints for this simple implementation
    if (TILE > 32) {
        printf("Note: TILE > 32 is unusual for transpose. Try TILE=16 or 32.\n");
    }

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s\n", prop.name);

    printf("=== Transpose benchmark ===\n");
    printf("N=%d, TILE=%d, iters=%d\n", N, TILE, iters);

    size_t bytes = (size_t)N * (size_t)N * sizeof(float);

    float* hA = (float*)std::malloc(bytes);
    float* hB = (float*)std::malloc(bytes);
    if (!hA || !hB) {
        fprintf(stderr, "Host malloc failed (try smaller N)\n");
        return 1;
    }
    init_matrix(hA, N);

    float *dA=nullptr, *dB=nullptr;
    CUDA_CHECK(cudaMalloc(&dA, bytes));
    CUDA_CHECK(cudaMalloc(&dB, bytes));
    CUDA_CHECK(cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice));

    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE,
              (N + TILE - 1) / TILE);

    int warmup = 10;

    // Naive
    float ms_naive = time_kernel_ms_naive(dA, dB, N, grid, block, warmup, iters);
    CUDA_CHECK(cudaMemcpy(hB, dB, bytes, cudaMemcpyDeviceToHost));
    float err_naive = max_abs_err_transpose(hA, hB, N);

    // Tiled
    float ms_tiled = time_kernel_ms_tiled(dA, dB, N, TILE, grid, block, warmup, iters);
    CUDA_CHECK(cudaMemcpy(hB, dB, bytes, cudaMemcpyDeviceToHost));
    float err_tiled = max_abs_err_transpose(hA, hB, N);

    // Bandwidth model: transpose reads 1 element + writes 1 element = 2 * bytes total
    double moved = 2.0 * (double)bytes;
    double gbps_naive = (moved / (ms_naive / 1e3)) / 1e9;
    double gbps_tiled = (moved / (ms_tiled / 1e3)) / 1e9;

    printf("block=(%d,%d) grid=(%d,%d)\n", block.x, block.y, grid.x, grid.y);
    printf("naive: avg=%.6f ms  approx=%.2f GB/s  max_err=%g\n", ms_naive, gbps_naive, err_naive);
    printf("tiled: avg=%.6f ms  approx=%.2f GB/s  max_err=%g\n", ms_tiled, gbps_tiled, err_tiled);

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    std::free(hA);
    std::free(hB);
    return 0;
}
