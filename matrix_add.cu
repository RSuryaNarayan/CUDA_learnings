// square matrix addition
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

__global__ void matrix_add (float *d_A, float *d_B, float *d_S, int N)
{
    /*
    In 2D, we have a 2D grid of blocks, each block is a "tile" of 2D threads. Inside a block,
    we get the thread index using threadIdx.x and threadIdx.y, block Id itself using
    blockIdx.x and blockIdx.y. So just extend everything in 1D to 2D!. Another important 2D note:
    threads are naturally continuous in a warp **across columns** i.e. they walk across colums (i.e. x)
    So you want to linearize the arrays in *row major order* to ensure contiguous memory lanes in a warp
    */
    // This would be the naive implementation without grid striding
    /*int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col < N && row < N)
    {
        int idx = row * N + col;
        d_S[idx] = d_A[idx] + d_B[idx];
    }*/

    // using grid strided loops to ensure coverage:
    for (int col = blockIdx.x * blockDim.x + threadIdx.x; col < N; col += blockDim.x * gridDim.x)
    {
        for (int row = blockIdx.y * blockDim.y + threadIdx.y; row < N; row+= blockDim.y * gridDim.y)
        {
            int idx = row * N + col;
            d_S[idx] = d_A[idx] + d_B[idx];
        }
    }
}

int main()
{
    //allocate and declare host
    int N=3;
    float h_A[3][3] = {{1.0f, 2.0f, 3.0f},{1.0f, 2.0f, 3.0f},{1.0f, 2.0f, 3.0f}};
    float h_B[3][3] = {{1.0f, 2.0f, 3.0f},{1.0f, 2.0f, 3.0f},{1.0f, 2.0f, 3.0f}};
    float h_S[3][3] = {{0.0f}};

    //allocate device
    float *d_A=nullptr, *d_B=nullptr, *d_S=nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, (size_t) N*N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, (size_t) N*N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_S, (size_t) N*N*sizeof(float)));

    //memcpys
    CUDA_CHECK(cudaMemcpy(d_A, h_A, (size_t)N*N*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, (size_t)N*N*sizeof(float), cudaMemcpyHostToDevice));

    //initialize grid/block dim and call kernel
    dim3 threads_per_block(3,3);
    dim3 blocks_in_grid(1,1);
    matrix_add<<<blocks_in_grid,threads_per_block>>>(d_A, d_B, d_S, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    //memcpy back
    CUDA_CHECK(cudaMemcpy(h_S, d_S, (size_t)N*N*sizeof(float), cudaMemcpyDeviceToHost));

    //print
    for (int i=0; i<N; i++)
    {
        for (int j=0; j<N; j++)
        {
            printf("%2.5f ", h_S[i][j]);
        }
        printf("\n");
    }

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_S));

    return 0;
}
