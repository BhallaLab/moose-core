#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <algorithm>

#include "CudaGlobal.h"

#include "RateLookup.h"
#include "HSolveActive.h"

#ifdef USE_CUDA

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/system/system_error.h>
#include <thrust/copy.h>

__device__ __constant__ int instant_xyz_d[3];

/*
 * Copy row arrays to device.
 * To isolate CUDA functions from HSolveActive.cpp
 */
void HSolveActive::copy_to_device(double ** v_row_array, double * v_row_temp, int size)
{
	cudaSafeCall(cudaMalloc((void**)v_row_array, sizeof(double) * size));
	cudaSafeCall(cudaMemcpy(*v_row_array, v_row_temp, sizeof(double) * size, cudaMemcpyHostToDevice));
}


/*
 * The kernel function to be executed on each CUDA thread.
 * 
 * This version uses one thread for one channel. 
 */
__global__
void advanceChannel_kernel(
	double                          * vTable,
	const unsigned                  v_nColumns,
	double							* v_row_array,
	LookupColumn                    * column_array,                      
	double                          * caTable,
	const unsigned                  ca_nColumns,
	ChannelData 					* channel,
	double                           * ca_row_array,
	double                          * istate,
	const unsigned                  channel_size,
	double                          dt,
	const unsigned					num_of_compartment
	)
{
	int tID = threadIdx.x + blockIdx.x * blockDim.x;
	int id = tID;
	if ((tID)>= channel_size) return;

	//Load channel info into thread local memory.
	u64 data = channel[tID];	

	tID = get_state_index(data);
	double myrow = v_row_array[get_compartment_index(data)];		
	double * iTable;
	unsigned inCol;
	
	bool xyz[3] = {get_x(data), get_y(data), get_z(data)};

	for(int i = 0; i < 3; ++i)
	{	
		if(!xyz[i]) continue;

		if (i == 2 && ca_row_array[get_ca_row_index(data)]!= -1.0f){
			myrow = ca_row_array[get_ca_row_index(data)];
			iTable = caTable;
			inCol = ca_nColumns;				
		}
		else {
			iTable = vTable;
			inCol = v_nColumns;
		}
		
		double a,b,C1,C2;
		double *ap, *bp;

		ap = iTable + int(myrow) + column_array[tID].column;
		bp = ap + inCol;	
		a = *ap;	
		b = *bp;
		
		C1 = a + ( b - a ) * (myrow - int(myrow));
		
		a = *( ap + 1 );		
		b = *( bp + 1 );	

		C2 = a + ( b - a ) * (myrow - int(myrow));

		/*
		 *instant_xyz_d is a CudaSymbol defined in copy_data.
		 *This array is kept in device memory as a global 
		 *constant array that can be accessed from all kernels.
		 */
		if(get_instant(data) & instant_xyz_d[i]) {
			istate[tID + i] = C1 / C2;		
		}
		
		else{
			double temp = 1.0 + dt / 2.0 * C2;
			istate[tID] = ( istate[tID] * ( 2.0 - temp ) + dt * C1 ) / temp;			
		} 
		tID ++;
	} 
}


/*
 * Copy static data from host to device,
 */
void HSolveActive::copy_data(std::vector<LookupColumn>& column,
							 LookupColumn ** 			column_dd,
							 int * 						is_inited,
							 vector<ChannelData>&		channel_data,
							 ChannelData ** 			channel_data_dd,
							 const int 					x,
							 const int 					y,
							 const int 					z)
{
	//Check if copied already.
	if(!(*is_inited))
	{
		*is_inited = 1;
		int size = column.size();
		printf("column size is :%d.\n", size);

		cudaSafeCall(cudaMalloc((void**)column_dd, size * sizeof(LookupColumn)));
		cudaSafeCall(cudaMemcpy(*column_dd,
								&(column.front()),
								size * sizeof(LookupColumn),
								cudaMemcpyHostToDevice));
		cudaSafeCall(cudaMalloc((void**)channel_data_dd, channel_data.size() * sizeof(ChannelData)));
		cudaSafeCall(cudaMemcpy(*channel_data_dd,
								&(channel_data.front()),
								channel_data.size() * sizeof(ChannelData),
								cudaMemcpyHostToDevice));
		const int xyz[3] = {x,y,z};
		cudaSafeCall(cudaMemcpyToSymbol(instant_xyz_d, xyz, sizeof(int)*3, 0, cudaMemcpyHostToDevice));
	}	
}

/*
 * Driver function for advanceChannel calculation kernels.
 */
void HSolveActive::advanceChannel_gpu(
	double *						 v_row_d,
	vector<double>&               	 caRow,
	LookupColumn 					* column,                                           
	LookupTable&                     vTable,
	LookupTable&                     caTable,                       
	double                          * istate,
	ChannelData 					* channel,
	double                          dt,
	int 							set_size,
	int 							channel_size,
	int 							num_of_compartment
	)
{
	double * caRow_array_d;
	double * istate_d;

	int caSize = caRow.size();
	
	cudaEvent_t mem_start, mem_stop;
	float mem_elapsed;
	cudaEventCreate(&mem_start);
	cudaEventCreate(&mem_stop);

	cudaEventRecord(mem_start);

	cudaSafeCall(cudaMalloc((void **)&caRow_array_d, 		caRow.size() * sizeof(double)));  
	cudaSafeCall(cudaMalloc((void **)&istate_d, 			set_size * sizeof(double)));   

	cudaSafeCall(cudaMemcpy(caRow_array_d, &caRow.front(), sizeof(double) * caRow.size(), cudaMemcpyHostToDevice));
	cudaSafeCall(cudaMemcpy(istate_d, istate, set_size*sizeof(double), cudaMemcpyHostToDevice));
	

	//Copy static info of vTable.
	//Will only be executed once.
	if(!vTable.is_set())
	{
		vTable.set_is_set(true);
		vTable.copy_table();
	}

	//Copy static info of caTable.
	//Will only be executed once.
	if(!caTable.is_set())
	{
		caTable.set_is_set(true);
		caTable.copy_table();	
	}

	cudaCheckError();
	cudaEventRecord(mem_stop);
	cudaEventSynchronize(mem_stop);
	cudaEventElapsedTime(&mem_elapsed, mem_start, mem_stop);

	//printf("GPU memory transfer time: %fms.\n", mem_elapsed);

	//Set kernel launch parameters.
	//BLOCK_WIDTH can be set in CudaGlobals.h
	dim3 gridSize(channel_size/BLOCK_WIDTH + 1, 1, 1);
	dim3 blockSize(BLOCK_WIDTH,1,1); 

	if(channel_size <= BLOCK_WIDTH)
	{
		gridSize.x = 1;
		blockSize.x = channel_size; 
	}    

	//Launch CUDA kernel.
	advanceChannel_kernel<<<gridSize,blockSize>>>( 
		vTable.get_table_d(),
		vTable.get_num_of_columns(),
		v_row_d,
		column,
		caTable.get_table_d(),
		caTable.get_num_of_columns(),
		channel,
		caRow_array_d,
		istate_d,
		channel_size,
		dt,
		num_of_compartment
	);

	cudaCheckError(); 

	//Copy the result from device memory back to host.
	cudaSafeCall(cudaMemcpy(istate, istate_d, set_size * sizeof(double), cudaMemcpyDeviceToHost));

	cudaSafeCall(cudaDeviceSynchronize());    
 
 	cudaSafeCall(cudaFree(v_row_d));
	cudaSafeCall(cudaFree(caRow_array_d));
	cudaSafeCall(cudaFree(istate_d));
}
#endif
