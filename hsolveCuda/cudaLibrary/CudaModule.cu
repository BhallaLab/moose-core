/**
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */
#include "CudaModule.h"

#include <stdio.h>
#include <stdlib.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>


#define MAX_THREADS_IN_BLOCK 512;

float * device_fullMatrix;
float * device_V;
float * device_VMid; //Internal Calculation
float * device_Cm;
float * device_Em;
float * device_Rm;
float * device_Ra;
float * device_B;

unsigned int nCell, nComp;

unsigned int threadDimNumber, blockDimNumber;
/**
 * This macro checks return value of the CUDA runtime call and exits
 * the application if the call failed.
 */
#define CUDA_CHECK_RETURN(value) {											\
	cudaError_t _m_cudaStat = value;										\
	if (_m_cudaStat != cudaSuccess) {										\
		fprintf(stderr, "Error %s at line %d in file %s\n",					\
				cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
		exit(1);															\
	} }

/**
 * CUDA kernel function that reverses the order of bits in each element of the array.
 */
__global__ void kernel_updateFullMatrix(
		float * device_fullMatrix,
		float * B,
		float * V,
		float * Cm,
		float * Em,
		float * Rm,
		float dt,
		unsigned int nComp
		) {
	//TODO: fix memory usage matter

	unsigned int t = threadIdx.x;
	unsigned int baseIndex = t*nComp;

	unsigned int i;
	for ( i = 0; i < nComp; i++ )
	{
		unsigned int myIndex=baseIndex+i;
		B[myIndex  ] =
			V[ myIndex] * Cm[myIndex] 	/ ( dt / 2.0 ) +
							Em[ myIndex] / Rm[myIndex];
	}
	__syncthreads();
}

__global__ void kernel_forwardElimination(
		float * fullMatrix,
		float * B,
		unsigned int nComp
		) {
	unsigned int t = threadIdx.x;
	unsigned int baseIndex = t*nComp*nComp;

	unsigned int i,j,k;
	for ( i = 0; i < nComp - 1; i++ )
		for ( j = i + 1; j < nComp; j++ ) {
			double div = fullMatrix[baseIndex+ j*nComp+i ] / fullMatrix[baseIndex+ i*nComp+ i ];
			for ( k = 0; k < nComp; k++ )
				fullMatrix[ baseIndex+j*nComp+k ] -= div * fullMatrix[baseIndex+ i *nComp+ k ];
			B[ baseIndex+j ] -= div * B[ baseIndex+i ];
		}
	__syncthreads();
}
__global__ void kernel_backwardSubstitute(
		float * matrix,
		float * B,
		float * V,
		float * VMid,
		unsigned int nComp
		) {
	unsigned int t = threadIdx.x;
	unsigned int baseIndexMatrix = t*nComp*nComp;
	unsigned int baseIndex = t*nComp;

	unsigned int i,j,k;
	for ( i = nComp - 1; i >= 0; i-- ) {
		VMid[ baseIndex+ i ] = B[ baseIndex+ i ];

		for ( j = nComp - 1; j > i; j-- )
			VMid[ baseIndex+ i ] -= VMid[ baseIndex+ j ] * matrix[baseIndexMatrix+ i  *nComp+ j ];

		VMid[ baseIndex+ i ] /= matrix[baseIndexMatrix+ i *nComp+ i ];

		V[ baseIndex+ i ] = 2 * VMid[ baseIndex+ i ] - V[ baseIndex+ i ];
	}

	__syncthreads();
}

/**
 * Host function that prepares data array and passes it to the CUDA kernel.
 */
int cudaModule_setup(
		const unsigned int __cellNumber,
		const unsigned int __nComp,
		float * host_fullMatrix,
		float * host_V,
		float * host_Cm,
		float * host_Em,
		float * host_Rm,
		float * host_Ra) {

	nCell = __cellNumber;
	nComp = __nComp;
	// Allocate memory
	CUDA_CHECK_RETURN(
			cudaMalloc((void **) &device_fullMatrix, __nComp * __nComp * __cellNumber * sizeof(float)));
	CUDA_CHECK_RETURN(
			cudaMalloc((void **) &device_Cm, __nComp * __cellNumber * sizeof(float)));
	CUDA_CHECK_RETURN(
			cudaMalloc((void **) &device_Em, __nComp * __cellNumber * sizeof(float)));
	CUDA_CHECK_RETURN(
			cudaMalloc((void **) &device_Ra, __nComp * __cellNumber * sizeof(float)));
	CUDA_CHECK_RETURN(
			cudaMalloc((void **) &device_Rm, __nComp * __cellNumber * sizeof(float)));
	CUDA_CHECK_RETURN(
			cudaMalloc((void **) &device_V, __nComp * __cellNumber * sizeof(float)));
	CUDA_CHECK_RETURN(
			cudaMalloc((void **) &device_B, __nComp * __cellNumber * sizeof(float)));

	CUDA_CHECK_RETURN(
				cudaMalloc((void **) &device_VMid, __nComp * __cellNumber * sizeof(float)));


	//Copy data
	CUDA_CHECK_RETURN(
			cudaMemcpy(device_fullMatrix, host_fullMatrix,__nComp * __nComp * __cellNumber * sizeof(float), cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(
			cudaMemcpy(device_Cm, host_Cm,__nComp * __cellNumber * sizeof(float), cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(
			cudaMemcpy(device_Em, host_Em,__nComp * __cellNumber * sizeof(float), cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(
			cudaMemcpy(device_Ra, host_Ra,__nComp * __cellNumber * sizeof(float), cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(
			cudaMemcpy(device_Rm, host_Rm,__nComp * __cellNumber * sizeof(float), cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(
			cudaMemcpy(device_V, host_V,__nComp * __cellNumber * sizeof(float), cudaMemcpyHostToDevice));

	// Initialize the grid and block dimensions
	//TODO: Consider changes in different numbers
	threadDimNumber = __cellNumber;
	blockDimNumber = 1;

	return 0;
}

void cudaModule_updateMatrix(float dt)
{
	dim3 threadDim (threadDimNumber);
	dim3 blockDim (blockDimNumber);

	kernel_updateFullMatrix <<<threadDim, blockDim>>>
	(device_fullMatrix, device_B,device_V, device_Cm, device_Em, device_Rm, dt,nComp);
}

void cudaMosule_forwardElimination()
{
	dim3 threadDim (threadDimNumber);
		dim3 blockDim (blockDimNumber);

	kernel_forwardElimination<<<threadDim, blockDim>>>
	(device_fullMatrix, device_B,nComp);
}

void cudaMosule_backwardSubstitute()
{
	dim3 threadDim (threadDimNumber);
			dim3 blockDim (blockDimNumber);

	kernel_backwardSubstitute<<<threadDim, blockDim>>>
		(device_fullMatrix, device_B,device_V,device_VMid,nComp);
}
int cudaModule_getB(float * B)
{
	CUDA_CHECK_RETURN(
			cudaMemcpy(B,device_B,nComp * nCell * sizeof(float), cudaMemcpyDeviceToHost));
	return 0;
}

void cudaModule_discard()
{
	cudaFree(device_fullMatrix);
	cudaFree(device_Cm);
	cudaFree(device_Em);
	cudaFree(device_Ra);
	cudaFree(device_Rm);
	cudaFree(device_V);
	cudaFree(device_VMid);
	cudaFree(device_B);
}

//TEST

static const int WORK_SIZE = 256;

#define CUDA_CHECK_RETURN(value) {											\
	cudaError_t _m_cudaStat = value;										\
	if (_m_cudaStat != cudaSuccess) {										\
		fprintf(stderr, "Error %s at line %d in file %s\n",					\
				cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
		exit(1);															\
	} }

__device__ unsigned int bitreverse(unsigned int number) {
	number = ((0xf0f0f0f0 & number) >> 4) | ((0x0f0f0f0f & number) << 4);
	number = ((0xcccccccc & number) >> 2) | ((0x33333333 & number) << 2);
	number = ((0xaaaaaaaa & number) >> 1) | ((0x55555555 & number) << 1);
	return number;
}

/**
 * CUDA kernel function that reverses the order of bits in each element of the array.
 */
__global__ void bitreverse(void *data) {
	unsigned int *idata = (unsigned int*) data;
	idata[threadIdx.x] = bitreverse(idata[threadIdx.x]);
}

int cudaModule_test()
{
	printf("CUDA Clock sample\n");
	void *d = NULL;
	int i;
	unsigned int idata[WORK_SIZE], odata[WORK_SIZE];

	for (i = 0; i < WORK_SIZE; i++)
		idata[i] = (unsigned int) i;

	CUDA_CHECK_RETURN(cudaMalloc((void**) &d, sizeof(int) * WORK_SIZE));
	CUDA_CHECK_RETURN(
			cudaMemcpy(d, idata, sizeof(int) * WORK_SIZE, cudaMemcpyHostToDevice));

	bitreverse<<<1, WORK_SIZE, WORK_SIZE * sizeof(int)>>>(d);

	CUDA_CHECK_RETURN(cudaThreadSynchronize());	// Wait for the GPU launched work to complete
	CUDA_CHECK_RETURN(cudaGetLastError());
	CUDA_CHECK_RETURN(cudaMemcpy(odata, d, sizeof(int) * WORK_SIZE, cudaMemcpyDeviceToHost));

	for (i = 0; i < WORK_SIZE; i++)
		printf("Input value: %u, device output: %u\n", idata[i], odata[i]);

	CUDA_CHECK_RETURN(cudaFree((void*) d));
	CUDA_CHECK_RETURN(cudaDeviceReset());
	printf("CUDA Test Done. \n");
	return 0;
}

