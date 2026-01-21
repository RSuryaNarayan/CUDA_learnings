// vecadd_bench.cu
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

// Kernel assumes U is stored in a Structure of Arrays (SoA) fashion i.e.
// U = [u1, u2, u3, ..., uNcells, v1, v2, v3, ..., vNcells, w1, w2, w3, ..., wNcells]
__global__ void magSqrU_SoA (float* U_SoA, float* magSqrU, int Ncells)
{
    for (int idx = blockDim.x * blockIdx.x + threadIdx.x;
         idx < Ncells;
        idx += gridDim.x * blockDim.x)
    {
        float u = U_SoA[idx + 0*Ncells];
        // float v = U_SoA[idx + 1*Ncells];
        // float w = U_SoA[idx + 2*Ncells];

        magSqrU[idx] = u * u; 
        //             + v * v + w * w;
        /* This is another way to write things to avoid register pressure
           by declaring 3 floats above. But in this case, its probably a 
           tiny overhead and should be OK. */
        // magSqrU[idx] = U_SoA[idx + 0*Ncells] * U_SoA[idx + 0*Ncells] + 
        //                U_SoA[idx + 1*Ncells] * U_SoA[idx + 1*Ncells] +
        //                U_SoA[idx + 2*Ncells] * U_SoA[idx + 2*Ncells];
    }

    /*
     Note above that between consecutive threads in a warp, 
     we get contiguous memory access patterns in this way! i.e.
     thread 0 => magSqrU[0] = U_SoA[0]^2 + U_SoA[Ncells]^2 + U_SoA[2*Ncells]^2 
     thread 1 => magSqrU[1] = U_SoA[1]^2 + U_SoA[1 + Ncells]^2 + U_SoA[1 + 2*Ncells]^2 
    */
}

// Kernel assumes U is stored in a Array of Structures (AoS) fashion i.e.
// U = [u1, v1, w1, | u2, v2, w2, |..., | uNcells, vNcells, wNcells]
__global__ void magSqrU_AoS (float* U_AoS, float* magSqrU, int Ncells)
{
    for (int idx = blockDim.x * blockIdx.x + threadIdx.x;
        idx < Ncells;
        idx+= blockDim.x * gridDim.x)
    {
        float u = U_AoS[3*idx + 0];
        // float v = U_AoS[3*idx + 1];
        // float w = U_AoS[3*idx + 2];

        magSqrU[idx] = u * u;
        //            + v * v + w * w;
    }
    /*
     Note above that between consecutive threads in a warp, 
     we get jumps/strides of 3 i.e.
     thread 0 => magSqrU[0] = U_AoS[0]^2 + U_AoS[1]^2 + U_AoS[2]^2 
     thread 1 => magSqrU[1] = U_AoS[3]^2 + U_AoS[4]^2 + U_AoS[5]^2 
     This is pretty bad for performance within a warp as discontinuous memory locations 
     trigger a greater number of transactions to fetch data from main memory through the cache lines
    */
}

static void init_arrays(float* u, float* v, float* w, int Ncells)
{
    for (int i = 0; i < Ncells; i++) {
        u[i] = (float)(i % 97) * 0.01f;
        v[i] = (float)(i % 89) * 0.01f;
        w[i] = (float)(i % 109) * 0.01f;
    }
}

static float max_abs_err_magSqrU(const float* u, const float* v, const float* w, const float* magSqrU, int Ncells)
{
    float max_abs = 0.0f;
    for (int i = 0; i < Ncells; i++) {
        float ref = u[i]*u[i]; 
                    //+ v[i]*v[i] + w[i]*w[i];
        max_abs = fmaxf(max_abs, fabsf(magSqrU[i] - ref));
    }
    return max_abs;
}

static float time_magSqrU_SoA_ms( float* d_U_SoA, float* d_magSqrU_SoA, 
                            int Ncells, int grid, int block,
                            int warmup, int iters)
{
    // warmup (also catches runtime execution errors early)
    for (int i = 0; i < warmup; i++) {
        magSqrU_SoA<<<grid, block>>>(d_U_SoA, d_magSqrU_SoA, Ncells);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int k = 0; k < iters; k++) {
        magSqrU_SoA<<<grid, block>>>(d_U_SoA, d_magSqrU_SoA, Ncells);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaGetLastError()); // launch errors (timed region still valid)

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return ms / iters;
}

static float time_magSqrU_AoS_ms( float* d_U_AoS, float* d_magSqrU_AoS, 
                            int Ncells, int grid, int block,
                            int warmup, int iters)
{
    // warmup (also catches runtime execution errors early)
    for (int i = 0; i < warmup; i++) {
        magSqrU_AoS<<<grid, block>>>(d_U_AoS, d_magSqrU_AoS, Ncells);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int k = 0; k < iters; k++) {
        magSqrU_AoS<<<grid, block>>>(d_U_AoS, d_magSqrU_AoS, Ncells);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaGetLastError()); // launch errors (timed region still valid)

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return ms / iters;
}

// U is stored in a Structure of Arrays (SoA) fashion i.e.
// U = [u1, u2, u3, ..., uNcells, v1, v2, v3, ..., vNcells, w1, w2, w3, ..., wNcells]
static void unroll_as_SoA (const float* u, const float* v, const float* w, float* U_SoA, const int Ncells)
{
    for (int i=0; i<Ncells; i++)
    {
        U_SoA[i + 0*Ncells] = u[i];
        U_SoA[i + 1*Ncells] = v[i];
        U_SoA[i + 2*Ncells] = w[i];
    }
}

// U is stored in a Array of Structures (AoS) fashion i.e.
// U = [u1, v1, w1, | u2, v2, w2, |..., | uNcells, vNcells, wNcells]
static void unroll_as_AoS (const float* u, const float* v, const float* w, float* U_AoS, const int Ncells)
{
    for (int i=0; i<Ncells; i++)
    {
        U_AoS[3*i + 0] = u[i];
        U_AoS[3*i + 1] = v[i];
        U_AoS[3*i + 2] = w[i];
    }
}

int main()
{
    {
        // Keep a tiny correctness test ALWAYS.
        const int Ncells_small = 6;
        float h_u[Ncells_small] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
        float h_v[Ncells_small] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
        float h_w[Ncells_small] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

        float* h_U_AoS = (float*) std::malloc((size_t)3*Ncells_small* sizeof(float));
        float* h_U_SoA = (float*) std::malloc((size_t)3*Ncells_small* sizeof(float));
        float* h_magSqrU_SoA = (float*) std::malloc((size_t)Ncells_small* sizeof(float));
        float* h_magSqrU_AoS = (float*) std::malloc((size_t)Ncells_small* sizeof(float));

        unroll_as_SoA(h_u, h_v, h_w, h_U_SoA, Ncells_small);
        unroll_as_AoS(h_u, h_v, h_w, h_U_AoS, Ncells_small);

        printf("======Small Ncells Test (Ncells=6)========");
        printf("\n===============SoA======================\n");
        for (int i=0; i<3*Ncells_small; i++)
        {
            printf("\nU_SoA[%d] = %2.5f", i, h_U_SoA[i]);
        }
        printf("\n===============AoS======================\n");
        for (int i=0; i<3*Ncells_small; i++)
        {
            printf("\nU_AoS[%d] = %2.5f", i, h_U_AoS[i]);
        }
        printf("\n=================Results====================\n");

        //declare and allocate device side pointers 
        float *d_U_SoA = nullptr, *d_U_AoS = nullptr, *d_magSqrU_SoA = nullptr, *d_magSqrU_AoS = nullptr;
        CUDA_CHECK(cudaMalloc(&d_U_SoA, (size_t)3*Ncells_small*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_U_AoS, (size_t)3*Ncells_small*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_magSqrU_SoA, (size_t)Ncells_small*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_magSqrU_AoS, (size_t)Ncells_small*sizeof(float)));

        //memcpys
        CUDA_CHECK(cudaMemcpy(d_U_SoA, h_U_SoA, (size_t)3*Ncells_small*sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_U_AoS, h_U_AoS, (size_t)3*Ncells_small*sizeof(float), cudaMemcpyHostToDevice));

        //launch kernels
        magSqrU_SoA<<<1,6>>>(d_U_SoA, d_magSqrU_SoA, Ncells_small);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        magSqrU_AoS<<<1,6>>>(d_U_AoS, d_magSqrU_AoS, Ncells_small);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        //memcpy back to host
        CUDA_CHECK(cudaMemcpy(h_magSqrU_SoA, d_magSqrU_SoA, (size_t)Ncells_small*sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_magSqrU_AoS, d_magSqrU_AoS, (size_t)Ncells_small*sizeof(float), cudaMemcpyDeviceToHost));

        // print and check
        printf("\n===============SoA Results======================\n");
        for (int i=0; i<Ncells_small; i++)
        {
            printf("\nmagSqrU_SoA[%d] = %2.5f", i, h_magSqrU_SoA[i]);
        }
        printf("\n===============AoS Results======================\n");
        for (int i=0; i<Ncells_small; i++)
        {
            printf("\nmagSqrU_AoS[%d] = %2.5f", i, h_magSqrU_AoS[i]);
        }
        printf("\n========================================\n");

        //free memory
        CUDA_CHECK(cudaFree(d_U_SoA));
        CUDA_CHECK(cudaFree(d_U_AoS));
        CUDA_CHECK(cudaFree(d_magSqrU_SoA));
        CUDA_CHECK(cudaFree(d_magSqrU_AoS));
        std::free(h_U_SoA);
        std::free(h_U_AoS);
        std::free(h_magSqrU_SoA);
        std::free(h_magSqrU_AoS);
    }
    
    printf("====Small Test Done=====");

    // Larger N test with timings
    int N_cells = 10000000;
    printf("\nN = %d\n", N_cells);
    for (int threads_per_block: {64, 128, 256, 512, 1024})
    {
        int blocks_on_grid = (N_cells + threads_per_block - 1)/threads_per_block;

        //allocate device side data
        float* h_u = (float*) std::malloc((size_t)N_cells* sizeof(float));
        float* h_v = (float*) std::malloc((size_t)N_cells* sizeof(float));
        float* h_w = (float*) std::malloc((size_t)N_cells* sizeof(float));
        float* h_U_SoA = (float*) std::malloc((size_t)3*N_cells* sizeof(float));
        float* h_U_AoS = (float*) std::malloc((size_t)3*N_cells* sizeof(float));
        float* h_magSqrU_SoA = (float*) std::malloc((size_t)N_cells* sizeof(float));
        float* h_magSqrU_AoS = (float*) std::malloc((size_t)N_cells* sizeof(float));
        init_arrays(h_u, h_v, h_w, N_cells);
        unroll_as_SoA(h_u, h_v, h_w, h_U_SoA, N_cells);
        unroll_as_AoS(h_u, h_v, h_w, h_U_AoS, N_cells);

        //allocate device side pointers
        float *d_U_SoA = nullptr, *d_U_AoS = nullptr, *d_magSqrU_AoS = nullptr, *d_magSqrU_SoA = nullptr;
        CUDA_CHECK(cudaMalloc(&d_U_SoA, (size_t)3*N_cells*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_U_AoS, (size_t)3*N_cells*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_magSqrU_SoA, (size_t)N_cells*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_magSqrU_AoS, (size_t)N_cells*sizeof(float)));

        //memcpys 
        CUDA_CHECK(cudaMemcpy(d_U_SoA, h_U_SoA, (size_t)3*N_cells*sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_U_AoS, h_U_AoS, (size_t)3*N_cells*sizeof(float), cudaMemcpyHostToDevice));

        //kernel launches
        magSqrU_SoA<<<blocks_on_grid, threads_per_block>>>(d_U_SoA, d_magSqrU_SoA, N_cells);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        magSqrU_AoS<<<blocks_on_grid, threads_per_block>>>(d_U_AoS, d_magSqrU_AoS, N_cells);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        //memcpys back
        CUDA_CHECK(cudaMemcpy(h_magSqrU_SoA, d_magSqrU_SoA, (size_t)N_cells*sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_magSqrU_AoS, d_magSqrU_AoS, (size_t)N_cells*sizeof(float), cudaMemcpyDeviceToHost));

        //error checking
        float max_err_SoA =0.0f, max_err_AoS =0.0f;
        max_err_SoA = max_abs_err_magSqrU(h_u, h_v, h_w, h_magSqrU_SoA, N_cells);
        max_err_AoS = max_abs_err_magSqrU(h_u, h_v, h_w, h_magSqrU_AoS, N_cells);

        //timing and bandwidth calculations
        float avg_kernel_time_SoA=0.0f, avg_kernel_time_AoS=0.0f;
        int warm_up = 100;
        int iters = 500;

        avg_kernel_time_SoA = time_magSqrU_SoA_ms( d_U_SoA, d_magSqrU_SoA, 
                                N_cells, blocks_on_grid, threads_per_block,
                                warm_up, iters);
        
        avg_kernel_time_AoS = time_magSqrU_AoS_ms( d_U_AoS, d_magSqrU_AoS, 
                                N_cells, blocks_on_grid, threads_per_block,
                                warm_up, iters);

        const double bytes_per_element = 2 * sizeof(float); // use 4 if using all 3 components 

        double gbytes_per_sec_SoA = (double)N_cells * bytes_per_element / (avg_kernel_time_SoA/1e3) / 1e9;
        double gbytes_per_sec_AoS = (double)N_cells * bytes_per_element / (avg_kernel_time_AoS/1e3) / 1e9;

        printf("\n=====Block size: %d, Num Blocks: %d,======\n", threads_per_block, blocks_on_grid);
        printf("\nKernel Time: %2.5f ms (SoA), %2.5f ms (AoS)", avg_kernel_time_SoA, avg_kernel_time_AoS);
        printf("\nMemory Bandwidth: %2.5f GB/s (SoA), %2.5f GB/s (AoS)", gbytes_per_sec_SoA, gbytes_per_sec_AoS);
        printf("\nmax abs error: %2.5f (SoA), %2.5f (AoS)\n", max_err_SoA, max_err_AoS);
        
        //free stuff!
        CUDA_CHECK(cudaFree(d_U_SoA));
        CUDA_CHECK(cudaFree(d_U_AoS));
        CUDA_CHECK(cudaFree(d_magSqrU_SoA));
        CUDA_CHECK(cudaFree(d_magSqrU_AoS));
        std::free(h_u);
        std::free(h_v);
        std::free(h_w);
        std::free(h_U_SoA);
        std::free(h_U_AoS);
        std::free(h_magSqrU_SoA);
        std::free(h_magSqrU_AoS);
    }

   return 0;
}
