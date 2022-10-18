
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <random>
#include <stdio.h>
#include <vector>
#include <chrono>

__global__ void VectorAdditionGPU(float *c, const float *a, const float *b)
{
    size_t index = blockIdx.x * blockDim.x + threadIdx.x; // index calculation
    c[index] = a[index] + b[index]; // computation
}

std::vector<float> VectorAdditionCPU(const float* A, const float* B, size_t n)
{
    std::vector<float> C((n*n),0);
    for (auto i = 0; i < (n*n); ++i)
    {
        C[i] = A[i] + B[i];
    }
    return C;
}

int main()
{
    std::random_device rd;// seed generator
    std::mt19937 gen(rd());// quick pseudo rng

    size_t n = static_cast<size_t>(gen() % 990) + 10; // n - square root of the vectors' lenght
    printf("Vector dimensions are 1x%d\n", (n*n));
    size_t matrixSize = n * n * sizeof(float);
    float* host_A = (float*)malloc(matrixSize);// mem alloc for dynamic structure
    float* host_B = (float*)malloc(matrixSize);
    float* host_C = (float*)malloc(matrixSize);

    for (auto i = 0; i < n; ++i)// initial vectors value gen
    {
        for (auto j = 0; j < n; ++j)
        {
            host_A[i * n + j] = static_cast<float>(gen());
            host_B[i * n + j] = static_cast<float>(gen());
        }
    }

    float* dev_A = nullptr; // device(gpu) vars
    float* dev_B = nullptr;
    float* dev_C = nullptr;

    cudaError_t cudaStatus;

    cudaStatus = cudaMalloc((void**)&dev_A, matrixSize); //mem alloc for gpu
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc failed!");
    }
    cudaStatus = cudaMalloc((void**)&dev_B, matrixSize);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc failed!");
    }
    cudaStatus = cudaMalloc((void**)&dev_C, matrixSize);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc failed!");
    }

    dim3 blocksPerGrid = dim3(((n * n) / 1024 + 1)); // grid size calc
    dim3 threadsPerBlock = dim3(1024);

    cudaEvent_t start, stop; // for elapsed time calc (gpu)
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    cudaStatus = cudaMemcpy(dev_A, host_A, matrixSize, cudaMemcpyHostToDevice);// copy data to gpu
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy failed!");
    }
    cudaStatus = cudaMemcpy(dev_B, host_B, matrixSize, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy failed!");
    }

    VectorAdditionGPU<<<blocksPerGrid, threadsPerBlock>>> (dev_C, dev_A, dev_B);// perform calc

    cudaStatus = cudaMemcpy(host_C, dev_C, matrixSize, cudaMemcpyDeviceToHost);// copy results from gpu
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy failed! %d", cudaStatus);
    }
    cudaEventRecord(stop, 0);// stop the timer
    cudaEventSynchronize(stop);
    float ElapsedTime;
    cudaEventElapsedTime(&ElapsedTime, start, stop);
    printf("Time spent on GPU calculation: %.3f milliseconds\n", ElapsedTime);

    auto begin = clock();// for elapsed time calc (cpu)
    
    auto vector = VectorAdditionCPU(host_A, host_B, n);

    float cputime = 1000*(float)(clock() - begin)/CLOCKS_PER_SEC;

    printf("Time spent on CPU calculation: %.3f milliseconds\n", cputime);

    printf("Precision test started...\n");
    //checking if some of the gpu results differ from those computed by cpu and how many there are
    int coarsecount = 0;
    for (auto i = 0; i < (n*n); ++i)
    {
        if ((host_C[i] - vector[i]) != 0)
        {
          coarsecount++;
        }
     }
    printf("Precision test finished. Ammount of coarse numbers: %d\n", coarsecount);

    free(host_A);// free the memory
    free(host_B);
    free(host_C);
    cudaFree(dev_A);
    cudaFree(dev_B);
    cudaFree(dev_C);

    return 0;
}
