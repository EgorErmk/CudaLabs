
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <random>
#include "time.h"
#include <vector>

__global__ void MatrixMultiplicationGPU(const float* A, const float* B, float* C, size_t n)
{
	size_t C_index = blockIdx.x * blockDim.x + threadIdx.x;
	size_t A0_index = C_index/n;
	size_t B0_index = C_index - (A0_index*n);
	float sum = 0;
	for (auto temp = 0; temp < n; ++temp)
		sum += A[A0_index + temp] * B[B0_index + temp*n];
	C[C_index] = sum;
}

std::vector<std::vector<float>> MatrixMultiplicationCPU(const float* A,const float* B, size_t n)
{
	std::vector<std::vector<float>> C(n,std::vector<float>(n,0));
	float sum = 0;
	for (auto i = 0; i < n; ++i)
	{
		for (auto j = 0; j < n; ++j)
		{
			for (auto k = 0; k < n; ++k)
			{
				sum += A[i*n + k] * B[k*n + j];
			}
			C[i][j] = sum;
			sum = 0;
		}
	}
	return C;
}
int main()
{
	std::random_device rd;
	std::mt19937 gen(rd());

	size_t n = static_cast<size_t>(gen() % 1900) + 100;
	printf("Matrix dimentions are %dx%d\n", n, n);
	size_t matrixSize = n * n * sizeof(float);
	float *host_A = (float*)malloc(matrixSize);
	float *host_B = (float*)malloc(matrixSize);
	float *host_C = (float*)malloc(matrixSize);

	for (auto i = 0; i < n; ++i)
	{
		for (auto j = 0; j < n; ++j)
		{
			host_A[i*n + j] = static_cast<float>(gen());
			host_B[i*n + j] = static_cast<float>(gen());
		}
	}

	float* dev_A = nullptr;
	float* dev_B = nullptr;
	float* dev_C = nullptr;

	cudaError_t cudaStatus;

	cudaStatus = cudaMalloc((void**)&dev_A, matrixSize);
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
	
	dim3 blocksPerGrid = dim3(((n*n) / 1024 + 1));
	dim3 threadsPerBlock = dim3(((n / 1024) ? 1024 : n));

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	
	cudaStatus = cudaMemcpy(dev_A, host_A, matrixSize, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) 
	{
		fprintf(stderr, "cudaMemcpy failed!");
	}
	cudaStatus = cudaMemcpy(dev_B, host_B, matrixSize, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMemcpy failed!");
	}

	MatrixMultiplicationGPU<<<blocksPerGrid,threadsPerBlock>>>(dev_A, dev_B, dev_C, n);

	cudaStatus = cudaMemcpy(host_C, dev_C, matrixSize, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMemcpy failed! %d", cudaStatus);
	}

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float ElapsedTime;
	cudaEventElapsedTime(&ElapsedTime, start, stop);
	printf("Time spent on GPU calculation: %.3f milliseconds\n",ElapsedTime);
	

	clock_t begin, end;
	begin = clock();
	//std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

	auto vector = MatrixMultiplicationCPU(host_A, host_B, n);

	end = clock();
	//std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
	float cputime = ((float)end - (float)begin)/ CLOCKS_PER_SEC;
	printf("Time spent on CPU calculation: %.3f seconds\n", cputime);

	free(host_A);
	free(host_B);
	free(host_C);
	cudaFree(dev_A);
	cudaFree(dev_B);
	cudaFree(dev_C);

	return 0;
}
/*
Device name : NVIDIA GeForce GTX 1050
Total global memory : 4095 MB
Shared memory per block : 49152
Registers per block : 65536
Warp size : 32
Memory pitch : 2147483647
Max threads per block : 1024
Max threads dimensions : x = 1024, y = 1024, z = 64
Max grid size: x = 2147483647, y = 65535, z = 65535
Clock rate: 1493000
Total constant memory: 65536
Compute capability: 6.1
Texture alignment: 512
Device overlap: 1
Multiprocessor count: 5
Kernel execution timeout enabled: true
*/