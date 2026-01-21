#include<cstdio>
#include<cstdlib>
#include<cmath>
#include<cuda_runtime.h>

// my own re-write of vecadd_bench.cu example from chatGPT to enhance understanding
// Task: we want to write a CUDA kernel that:
// (1) adds two vectors of large sizes A, B into a vector C
// (2) this kernel must automatically adjust to any launch configuration (done)
// (3) Launch this kernel a few times, and then time it over 500 iterations to check kernel avg. time
// (4) compute the rough GPU memory bandwidth in GB/s 
// (5) compare and report 'correctness' using max-abs-err against a CPU-only benchmark (done)
// (6) while coding CUDA API calls or kernels use a boilerplate macro to check if the calls have succeeded. (done)
// (7) before doing a large N test use a small N test using dummy arrays (done)
// (8) Also sweep through block sizes 

// boilerplate macro for checking CUDA API calls
#define CUDA_CHECK(call) do {                                              \
    cudaError_t _err = (call);                                              \
    if (_err != cudaSuccess) {                                              \
        fprintf(stderr, "CUDA error %s:%d: %s (%d)\n",                      \
                __FILE__, __LINE__, cudaGetErrorString(_err), (int)_err);   \
        std::exit(1);                                                      \
    }                                                                       \
} while (0)

// CUDA kernel to add two vecs
__global__ void vec_add (float* d_A, float* d_B, float* d_C, int N)
{
    // thread idx = blockDim.x*blockIdx.x + threadIdx.x
    // num_threads T = gridDim.x * blockDim.x
    // grid-strided access pattern follows: t_id, t_id+T, t_id+2T, ..., and so on
    for (int idx = blockDim.x * blockIdx.x + threadIdx.x;
        idx < N;
        idx += gridDim.x * blockDim.x )
    {
        d_C[idx] = d_A[idx] + d_B[idx];
    }
}

//now write a function to do a CPU correctness check using host side arrays
float get_max_abs_error_cpu( const float* h_A, const float* h_B, const float* h_C, const int N)
{
    float max_abs_error = 0;
    for (int i=0; i<N; i++)
    {
        max_abs_error = fmaxf(max_abs_error, fabsf(h_A[i]+h_B[i]-h_C[i]));
    }
    return max_abs_error;
}

//now write a function to initialize large arrays h_A, h_B
void initialize_large_cpu_arrays( float* h_A, float* h_B, int N)
{
    for (int i=0; i<N; i++)
    {
        h_A[i] = (float) i * 0.75;
        h_B[i] = (float) i * 0.69;
    }
}

float timing_function (float* d_A, float* d_B, float* d_C, int N, int warm_up, int iters, int blocks_in_grid, int threads_per_block)
{
    //launch kernel warm_up iter times first
    for (int w=0; w<warm_up; w++)
    {
        vec_add<<<blocks_in_grid, threads_per_block>>>(d_A, d_B, d_C, N);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    float avg_time = 0.0f;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int it=0; it<iters; it++)
    {
        vec_add<<<blocks_in_grid, threads_per_block>>>(d_A, d_B, d_C, N);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventElapsedTime(&avg_time, start, stop));
    avg_time = avg_time/iters;

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return avg_time;
}

int main()
{
    // small arrays quick check
    {
        const int N_small = 6;
        float h_A[N_small] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
        float h_B[N_small] = {2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};
        float h_C[N_small] = {0.0f};

        float *d_A = nullptr,  *d_B = nullptr, *d_C = nullptr;
        CUDA_CHECK(cudaMalloc(&d_A, (size_t)N_small*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_B, (size_t)N_small*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_C, (size_t)N_small*sizeof(float)));

        CUDA_CHECK(cudaMemcpy(d_A, h_A, (size_t)N_small*sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, h_B, (size_t)N_small*sizeof(float), cudaMemcpyHostToDevice));

        vec_add<<<1, 6>>>(d_A, d_B, d_C, N_small);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(h_C, d_C, (size_t)N_small*sizeof(float), cudaMemcpyDeviceToHost));

        printf("====Quick small test checking (N_small=6)=====");
        for(int i=0; i<N_small; i++)
        {
            printf("\n C[%d] = (%2.5f + %2.5f)=%2.5f \n", i, h_A[i], h_B[i], h_C[i]);
        }

        CUDA_CHECK(cudaFree(d_A));
        CUDA_CHECK(cudaFree(d_B));
        CUDA_CHECK(cudaFree(d_C));

        float max_err = 0.0f;
        max_err = get_max_abs_error_cpu(h_A, h_B, h_C, N_small);
        printf("\n Max err between CPU and GPU computations = %2.5f\n", max_err);

    }

    printf("====Small Test Done=====");
    int N = 50000000;
    printf("\nN = %d\n", N);
    for (int threads_per_block: {64, 128, 256, 512, 1024})
    {
        int blocks_in_grid = (N + threads_per_block-1)/threads_per_block; //ceil trick to fit all threads in N

        float* h_A = (float*) std::malloc((size_t)N* sizeof(float));
        float* h_B = (float*) std::malloc((size_t)N* sizeof(float));
        float* h_C = (float*) std::malloc((size_t)N* sizeof(float));

        initialize_large_cpu_arrays(h_A, h_B, N);

        float *d_A = nullptr,  *d_B = nullptr, *d_C = nullptr;
        CUDA_CHECK(cudaMalloc(&d_A, (size_t)N*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_B, (size_t)N*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_C, (size_t)N*sizeof(float)));

        CUDA_CHECK(cudaMemcpy(d_A, h_A, (size_t)N*sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, h_B, (size_t)N*sizeof(float), cudaMemcpyHostToDevice));

        vec_add<<<blocks_in_grid, threads_per_block>>>(d_A, d_B, d_C, N);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(h_C, d_C, (size_t)N*sizeof(float), cudaMemcpyDeviceToHost));

        float max_err_large = 0.0f;
        max_err_large = get_max_abs_error_cpu(h_A, h_B, h_C, N);

        float avg_kernel_time=0.0f;
        int warm_up = 100;
        int iters = 500;
        avg_kernel_time = timing_function (d_A, d_B, d_C, N, warm_up, iters, blocks_in_grid, threads_per_block);

        const double bytes_per_element = 3 * sizeof(float);
        double gbytes_per_sec = (double)N * bytes_per_element / (avg_kernel_time/1e3) / 1e9;

        printf("Block size: %d, Num Blocks: %d, Kernel Time: %2.5f ms, Memory Bandwidth: %2.5f GB/s, max error = %2.5f\n", threads_per_block, blocks_in_grid, avg_kernel_time, gbytes_per_sec, max_err_large);
        
        //free stuff!
        CUDA_CHECK(cudaFree(d_A));
        CUDA_CHECK(cudaFree(d_B));
        CUDA_CHECK(cudaFree(d_C));
        std::free(h_A);
        std::free(h_B);
        std::free(h_C);
    }
    return 0;
}

