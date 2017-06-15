#include "CudaGlobal.h"
#ifdef USE_CUDA
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <algorithm>
#include <fstream>


#include "RateLookup.h"
#include "HSolveActive.h"


#include "Gpu_timer.h"

/*
 * Given a lookup , it lookups the elements in the lookup table and
 * stores row and fraction values in the memory.
 *
 * The reason it is separated because, each compartment most likely
 * have more than one channel. Once lookup is done for a given compartment,
 * it can be used for all channels of that compartment.
 */
__global__
void get_lookup_rows_and_fractions_cuda(
		double* lookups,
		double* table,
		double min, double max, double dx,
		int* rows, double* fracs,
		unsigned int nColumns, unsigned int size){

	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	if(tid < size){
		double x = lookups[tid];

		if ( x < min )
			x = min;
		else if ( x > max )
			x = max;

		double div = ( x - min ) / dx;
		unsigned int integer = ( unsigned int )( div );

		rows[tid] = integer*nColumns;
		fracs[tid] = div-integer;
	}
}

/*
 * Based on the near lookup value and fraction value, the function
 * interpolates the value and uses it to update appropriate state variables.
 * "indices" array is a subset of compartment id's which are
 * voltage dependent gate indices or Calcium dependent gate indices
 */
__global__
void advance_channels_opt_cuda(
		int* rows,
		double* fracs,
		double* table,
		int* indices,
		int* gate_to_comp,
		double* gate_values,
		int* gate_columns,
		int* state2chanId,
		int* chan_instants,
		unsigned int nColumns,
		double dt,
		int size
		){
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(tid < size){
		double a,b,C1,C2;
		int index, lookup_index, row_start_index, column;

		index = indices[tid];
		lookup_index = gate_to_comp[tid];
		row_start_index = rows[lookup_index];
		column = gate_columns[index];

		a = table[row_start_index + column];
		b = table[row_start_index + column + nColumns];

		C1 = a + (b-a)*fracs[lookup_index];

		a = table[row_start_index + column + 1];
		b = table[row_start_index + column + 1 + nColumns];

		C2 = a + (b-a)*fracs[lookup_index];

		if(!chan_instants[state2chanId[tid]]){
			a = 1.0 + dt/2.0 * C2; // reusing a
			gate_values[index] = ( gate_values[index] * ( 2.0 - a ) + dt * C1 ) / a;
		}
		else{
			gate_values[index] = C1/C2;
		}
	}
}


/*
 * Gbar*(x1^p1)*(x2^p2) ... (xn^pn) is computed for each channel
 */
__global__
void calculate_channel_currents_opt_cuda(double* d_gate_values,
		double* d_gate_powers,
		int* rowPtr,
		double* d_chan_modulation,
		double* d_chan_Gbar,
		//CurrentStruct* d_current_, // This structure corresponds to current_ DS in CPU.
		double* d_chan_Ek,
		double* d_chan_Gk,
		double* d_chan_GkEk,
		int size){
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(tid < size){
		double temp = d_chan_modulation[tid] * d_chan_Gbar[tid];
		for (int i = rowPtr[tid]; i < rowPtr[tid+1]; ++i) {
			temp *= pow(d_gate_values[i], d_gate_powers[i]);
		}

		//d_current_[tid].Gk = temp;
		//d_chan_Gk[tid] = temp;
		//d_chan_GkEk[tid] = temp*d_current_[tid].Ek;

		d_chan_Gk[tid] = temp;
		d_chan_GkEk[tid] = temp*d_chan_Ek[tid];
	}
}

/*
 * Work Per Thread Kernel. If there are N independent things to do, each thread does it in parallel.
 * It turns out that if work is small and load balance is great, this it is faster.
 */
__global__
void wpt_kernel(double* d_chan_Gk, double* d_chan_GkEk , int* d_chan_rowPtr,
		double* d_comp_Gksum, double* d_comp_GkEksum, int num_comp){
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	if(tid < num_comp){
		double sum1=0, sum2=0;
		int i;
		for (i = d_chan_rowPtr[tid]; i < d_chan_rowPtr[tid+1]; ++i) {
			sum1 += d_chan_Gk[i];
			sum2 += d_chan_GkEk[i];
		}
		d_comp_Gksum[tid] = sum1;
		d_comp_GkEksum[tid] = sum2;
	}
}

/*
 * Updates the matrix data structure d_HS_.
 * Case : MOOSE solver on CPU
 */
__global__
void update_matrix_kernel(double* d_V,
		double* d_HS_,
		double* d_comp_Gksum,
		double* d_comp_GkEksum,
		CompartmentStruct* d_compartment_,
		InjectStruct* d_inject_,
		double*	d_externalCurrent_,
		int size){

	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(tid < size){
		d_HS_[4*tid] = d_HS_[4*tid+2] + d_comp_Gksum[tid] + d_externalCurrent_[2*tid];
		d_HS_[4*tid+3] = d_V[tid]*d_compartment_[tid].CmByDt + d_compartment_[tid].EmByRm + d_comp_GkEksum[tid] +
				(d_inject_[tid].injectVarying + d_inject_[tid].injectBasal) + d_externalCurrent_[2*tid+1];

		d_inject_[tid].injectVarying = 0;
	}
}

/*
 * Updates the matrix data structure d_HS_.
 * Case : WPT approach + MOOSE Solver on CPU
 */
__global__
void update_matrix_kernel_opt(
		double* d_chan_Gk, double* d_chan_GkEk , int* d_chan_rowPtr,
		double* d_V,
		double* d_HS_,
		double* d_comp_Gksum,
		double* d_comp_GkEksum,
		CompartmentStruct* d_compartment_,
		InjectStruct* d_inject_,
		double*	d_externalCurrent_,
		int size){

	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(tid < size){

		double sum1=0, sum2=0;
		int i;
		for (i = d_chan_rowPtr[tid]; i < d_chan_rowPtr[tid+1]; ++i) {
			sum1 += d_chan_Gk[i];
			sum2 += d_chan_GkEk[i];
		}
		d_comp_Gksum[tid] = sum1;
		d_comp_GkEksum[tid] = sum2;

		d_HS_[4*tid] = d_HS_[4*tid+2] + sum1 + d_externalCurrent_[2*tid];
		d_HS_[4*tid+3] = d_V[tid]*d_compartment_[tid].CmByDt + d_compartment_[tid].EmByRm + sum2 +
				(d_inject_[tid].injectVarying + d_inject_[tid].injectBasal) + d_externalCurrent_[2*tid+1];

		//d_inject_[tid].injectVarying = 0;
	}
}

/*
 * Updates the matrix data structure d_perv_dynamic.
 * Case : WPT approach + Pervasive Solver on CPU.
 */
__global__
void update_perv_matrix_kernel_opt(
		double* d_chan_Gk, double* d_chan_GkEk , int* d_chan_rowPtr,
		double* d_V,
		double* d_perv_dynamic, double* d_perv_static,
		double* d_comp_Gksum,
		double* d_comp_GkEksum,
		CompartmentStruct* d_compartment_,
		//InjectStruct* d_inject_,
		double* d_stim_basal_values, int*  d_stim_map,
		double*	d_externalCurrent_,
		int size){

	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(tid < size){

		double sum1=0, sum2=0;
		int i;
		for (i = d_chan_rowPtr[tid]; i < d_chan_rowPtr[tid+1]; ++i) {
			sum1 += d_chan_Gk[i];
			sum2 += d_chan_GkEk[i];
		}
		d_comp_Gksum[tid] = sum1;
		d_comp_GkEksum[tid] = sum2;

		d_perv_dynamic[2*tid] = d_perv_static[tid] + sum1 + d_externalCurrent_[2*tid];
		//d_perv_dynamic[2*tid+1] = d_V[tid]*d_compartment_[tid].CmByDt + d_compartment_[tid].EmByRm + sum2 +
		//		(d_inject_[tid].injectVarying + d_inject_[tid].injectBasal) + d_externalCurrent_[2*tid+1];

		if(d_stim_map[tid] != -1)
			sum2 += d_stim_basal_values[d_stim_map[tid]]; // Adding currents of stimulated compartments.

		d_perv_dynamic[2*tid+1] = d_V[tid]*d_compartment_[tid].CmByDt + d_compartment_[tid].EmByRm + sum2 +
						// (d_inject_[tid].injectVarying + d_inject_[tid].injectBasal) +
						d_externalCurrent_[2*tid+1];

	}
}

/*
 * EXPERIMENTAL . Updates CSR matrix
 */
__global__
void update_csr_matrix_kernel(double* d_V,
				double* d_mat_values, double* d_main_diag_passive, int* d_main_diag_map, double* d_b,
				double* d_comp_Gksum,
				double* d_comp_GkEksum,
				CompartmentStruct* d_compartment_,
				InjectStruct* d_inject_,
				double*	d_externalCurrent_,
				int size){
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(tid < size){
		double main_val = d_main_diag_passive[tid] + d_comp_Gksum[tid] + d_externalCurrent_[2*tid];
		d_mat_values[d_main_diag_map[tid]] = main_val;
		d_b[tid] = d_V[tid]*d_compartment_[tid].CmByDt + d_compartment_[tid].EmByRm + d_comp_GkEksum[tid] +
				(d_inject_[tid].injectVarying + d_inject_[tid].injectBasal) + d_externalCurrent_[2*tid+1];

		d_inject_[tid].injectVarying = 0;
	}
}

/*
 * Kernel for calculating V(t+1) using Vmid(t) and V(t)
 */
__global__
void calculate_V_from_Vmid(double* d_Vmid, double* d_V, int size){
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if(tid < size){
		d_V[tid] = 2*d_Vmid[tid] - d_V[tid];
	}
}

/*
 * Calculating calcium currents.
 * Case : SPMV approach
 */
__global__
void advance_calcium_cuda(int* d_catarget_channel_indices,
			double* d_chan_Gk, double* d_chan_GkEk,
			double* d_Vmid,
			double* d_capool_values, int* d_chan_to_comp,
			int size){
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if(tid < size){
		int chan_id = d_catarget_channel_indices[tid];
		d_capool_values[tid] = d_chan_GkEk[chan_id] - d_chan_Gk[chan_id]*d_Vmid[d_chan_to_comp[chan_id]];
	}
}

/*
 * Advancing calcium pool in each time-step and clipping if necessary.
 * Case : SPMV approach
 */
__global__
void advance_calcium_conc_cuda(CaConcStruct* d_caConc_, double* d_Ca, double* d_caActivation_values, int size ){
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if(tid < size){
		d_caConc_[tid].c_ = d_caConc_[tid].factor1_ * d_caConc_[tid].c_ + d_caConc_[tid].factor2_ * d_caActivation_values[tid];
		double new_ca = d_caConc_[tid].CaBasal_ + d_caConc_[tid].c_;

		if(new_ca >  d_caConc_[tid].ceiling_){
			new_ca = d_caConc_[tid].ceiling_;
			d_caConc_[tid].c_ = new_ca - d_caConc_[tid].ceiling_;
		}
		if(new_ca < d_caConc_[tid].floor_){
			new_ca = d_caConc_[tid].floor_;
			d_caConc_[tid].c_ = new_ca - d_caConc_[tid].floor_;
		}
		d_Ca[tid] = new_ca;
	}
}

/* Calculating calcium currents.
 * Advancing calcium pool in each time-step and clipping if necessary.
 * Case : WPT approach.
 */
__global__
void advance_calcium_cuda_opt(int* d_catarget_channel_indices,
			double* d_chan_Gk, double* d_chan_GkEk,
			double* d_Vmid,
			//double* d_capool_values,
			int* d_chan_to_comp, int* rowPtr,
			//CaConcStruct* d_caConc_,
			double* d_Ca,
			double* d_CaConcStruct_c_, // Dynamic array
			double* d_CaConcStruct_CaBasal_, double* d_CaConcStruct_factor1_, double* d_CaConcStruct_factor2_, double* d_CaConcStruct_ceiling_, double* d_CaConcStruct_floor_, // Static array
			//double* d_caActivation_values,
			int size){
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if(tid < size){
		int chan_id;
		double sum = 0;
		for (int i = rowPtr[tid]; i < rowPtr[tid+1]; ++i) {
			chan_id = d_catarget_channel_indices[i];
			/*
			d_capool_values[i] = d_chan_GkEk[chan_id] - d_chan_Gk[chan_id]*d_Vmid[d_chan_to_comp[chan_id]];
			sum += d_capool_values[i];
			*/
			sum += d_chan_GkEk[chan_id] - d_chan_Gk[chan_id]*d_Vmid[d_chan_to_comp[chan_id]];
		}

		//d_caActivation_values[tid] = sum;
		/*
		//d_caConc_[tid].c_ = d_caConc_[tid].factor1_ * d_caConc_[tid].c_ + d_caConc_[tid].factor2_ * d_caActivation_values[tid];
		d_caConc_[tid].c_ = d_caConc_[tid].factor1_ * d_caConc_[tid].c_ + d_caConc_[tid].factor2_ * sum;
		double new_ca = d_caConc_[tid].CaBasal_ + d_caConc_[tid].c_;

		if(d_caConc_[tid].ceiling_ > 0 && new_ca >  d_caConc_[tid].ceiling_){
			new_ca = d_caConc_[tid].ceiling_;
			d_caConc_[tid].c_ = new_ca - d_caConc_[tid].CaBasal_;
		}
		if(new_ca < d_caConc_[tid].floor_){
			new_ca = d_caConc_[tid].floor_;
			d_caConc_[tid].c_ = new_ca - d_caConc_[tid].CaBasal_;
		}
		d_Ca[tid] = new_ca;
		*/

		d_CaConcStruct_c_[tid] = d_CaConcStruct_factor1_[tid]*d_CaConcStruct_c_[tid] + d_CaConcStruct_factor2_[tid]*sum;
		double new_ca = d_CaConcStruct_CaBasal_[tid] + d_CaConcStruct_c_[tid];

		if(d_CaConcStruct_ceiling_[tid] >0 && new_ca > d_CaConcStruct_ceiling_[tid]){
			new_ca = d_CaConcStruct_ceiling_[tid];
			d_CaConcStruct_c_[tid] = new_ca - d_CaConcStruct_CaBasal_[tid];
		}

		if(new_ca < d_CaConcStruct_floor_[tid]){
			new_ca = d_CaConcStruct_floor_[tid];
			d_CaConcStruct_c_[tid] = new_ca - d_CaConcStruct_CaBasal_[tid];
		}
		d_Ca[tid] = new_ca;
	}
}

//// CUDA Wrappers
// As GPU kernel cannot be called from CPP file, we have a wrapper function which does that.
void HSolveActive::get_lookup_rows_and_fractions_cuda_wrapper(double dt){

	int num_comps = V_.size();
	int num_Ca_pools = ca_.size();
	int num_cadep_gates = h_cagate_indices.size();

	int BLOCKS = num_comps/THREADS_PER_BLOCK;
	BLOCKS = (num_comps + THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK;

	// Getting lookup metadata for Vm
	get_lookup_rows_and_fractions_cuda<<<BLOCKS,THREADS_PER_BLOCK>>>(d_V,
    		d_V_table,
    		vTable_.get_min(), vTable_.get_max(), vTable_.get_dx(),
    		d_V_rows, d_V_fractions,
    		vTable_.get_num_of_columns(), num_comps);

	// Execute this block only if there are gates that are Ca dependent.
	if(num_cadep_gates > 0){
		// Getting lookup metadata for Ca
		BLOCKS = (num_Ca_pools + THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK;
		get_lookup_rows_and_fractions_cuda<<<BLOCKS,THREADS_PER_BLOCK>>>(d_ca,
				d_Ca_table,
				caTable_.get_min(), caTable_.get_max(), caTable_.get_dx(),
				d_Ca_rows, d_Ca_fractions,
				caTable_.get_num_of_columns(), num_Ca_pools);
	}

	#ifdef PIN_POINT_ERROR
		cudaCheckError(); // Checking for cuda related errors.
	#endif
}


void HSolveActive::advance_channels_cuda_wrapper(double dt){

	int num_vdep_gates = h_vgate_indices.size();
	int num_cadep_gates = h_cagate_indices.size();

	int BLOCKS = (num_vdep_gates+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK;
	// For Vm dependent gates
	advance_channels_opt_cuda<<<BLOCKS,THREADS_PER_BLOCK>>>(
			d_V_rows,
			d_V_fractions,
			d_V_table,
			d_vgate_indices,
			d_vgate_compIds,
			d_state_,
			d_state2column,
			d_state2chanId,
			d_chan_instant,
			vTable_.get_num_of_columns(),
			dt, num_vdep_gates );

	// Execute this block only if there are gates that are Ca dependent.
	if(num_cadep_gates > 0){
		BLOCKS = (num_cadep_gates+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK;
		// For Ca dependent gates.
		advance_channels_opt_cuda<<<BLOCKS,THREADS_PER_BLOCK>>>(
				d_Ca_rows,
				d_Ca_fractions,
				d_Ca_table,
				d_cagate_indices,
				d_cagate_capoolIds,
				d_state_,
				d_state2column,
				d_state2chanId,
				d_chan_instant,
				caTable_.get_num_of_columns(),
				dt, num_cadep_gates );
	}

	#ifdef PIN_POINT_ERROR
		cudaCheckError(); // Checking for cuda related errors.
	#endif
}


void HSolveActive::calculate_channel_currents_cuda_wrapper(){
	int num_channels = channel_.size();

	int BLOCKS = num_channels/THREADS_PER_BLOCK;
	BLOCKS = (num_channels%THREADS_PER_BLOCK == 0)?BLOCKS:BLOCKS+1; // Adding 1 to handle last threads

	calculate_channel_currents_opt_cuda<<<BLOCKS,THREADS_PER_BLOCK>>>(
				d_state_,
				d_state_powers,
				d_state_rowPtr,
				d_chan_modulation,
				d_chan_Gbar,
				//d_current_,
				d_chan_Ek,
				d_chan_Gk, d_chan_GkEk, num_channels);

#ifdef PIN_POINT_ERROR
	cudaCheckError(); // Checking for cuda related errors.
#endif
}

void HSolveActive::update_matrix_cuda_wrapper(){

	int num_channels = channel_.size();
	int BLOCKS;

	// As inject_ and externalCurrent_ data structures are updated by messages,
	// they have to be updated on the device too. Hence the transfer
	if(step_num%20 == 1)
		cudaMemcpy(d_inject_, &inject_[0], nCompt_*sizeof(InjectStruct), cudaMemcpyHostToDevice);

	cudaMemcpy(d_externalCurrent_, &(externalCurrent_.front()), 2 * nCompt_ * sizeof(double), cudaMemcpyHostToDevice);

	// As inject data is already on device, injectVarying can be set to zero.
	for (int i = 0; i < inject_.size(); ++i) {
		inject_[i].injectVarying = 0;
	}

	// Sending external current to GPU

	BLOCKS = (nCompt_+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK;
	if(UPDATE_MATRIX_APPROACH == UPDATE_MATRIX_WPT_APPROACH){
		// WPT approach for update matrix
		update_matrix_kernel_opt<<<BLOCKS,THREADS_PER_BLOCK>>>(d_chan_Gk, d_chan_GkEk , d_chan_rowPtr,
				d_V,
				d_HS_,
				d_comp_Gksum,
				d_comp_GkEksum,
				d_compartment_,
				d_inject_,
				d_externalCurrent_,
				(int)nCompt_);
	}else if(UPDATE_MATRIX_APPROACH == UPDATE_MATRIX_SPMV_APPROACH){
		// Using Cusparse
		const double alpha = 1.0;
		const double beta = 0.0;

		// SPMV approach for update matrix
		cusparseDcsrmv(cusparse_handle,  CUSPARSE_OPERATION_NON_TRANSPOSE,
			nCompt_, nCompt_, num_channels, &alpha, cusparse_descr,
			d_chan_Gk, d_chan_rowPtr, d_chan_colIndex,
			d_chan_x , &beta, d_comp_Gksum);

		cusparseDcsrmv(cusparse_handle,  CUSPARSE_OPERATION_NON_TRANSPOSE,
			nCompt_, nCompt_, num_channels, &alpha, cusparse_descr,
			d_chan_GkEk, d_chan_rowPtr, d_chan_colIndex,
			d_chan_x , &beta, d_comp_GkEksum);

		update_matrix_kernel<<<BLOCKS, THREADS_PER_BLOCK>>>(d_V,
						d_HS_,
						d_comp_Gksum,
						d_comp_GkEksum,
						d_compartment_,
						d_inject_,
						d_externalCurrent_,
						(int)nCompt_);
	}else{
		// Future approaches, if any.
	}

	cudaMemcpy(&HS_[0], d_HS_, HS_.size()*sizeof(double), cudaMemcpyDeviceToHost );
}

void HSolveActive::update_perv_matrix_cuda_wrapper(){

	int num_channels = channel_.size();
	int BLOCKS;

	// As inject_ and externalCurrent_ data structures are updated by messages,
	// they have to be updated on the device too. Hence the transfer
	if(step_num == 19){
		cudaMemcpy(d_stim_map, stim_map, nCompt_*sizeof(int), cudaMemcpyHostToDevice); // Initializing map.
	}
	if(step_num%20 == 1){
		cudaMemcpy(d_inject_, &inject_[0], nCompt_*sizeof(InjectStruct), cudaMemcpyHostToDevice);
		cudaMemcpy(d_stim_basal_values, stim_basal_values, num_stim_comp*sizeof(double), cudaMemcpyHostToDevice);
	}
	cudaMemcpy(d_externalCurrent_, &(externalCurrent_.front()), 2 * nCompt_ * sizeof(double), cudaMemcpyHostToDevice);

	// As inject data is already on device, injectVarying can be set to zero.
	for (int i = 0; i < inject_.size(); ++i) {
		inject_[i].injectVarying = 0;
	}

	// Sending external current to GPU

	BLOCKS = (nCompt_+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK;
	if(UPDATE_MATRIX_APPROACH == UPDATE_MATRIX_WPT_APPROACH){
		// WPT approach for update matrix
		update_perv_matrix_kernel_opt<<<BLOCKS,THREADS_PER_BLOCK>>>(d_chan_Gk, d_chan_GkEk , d_chan_rowPtr,
				d_V,
				d_perv_dynamic, d_perv_static,
				d_comp_Gksum,
				d_comp_GkEksum,
				d_compartment_,
				//d_inject_,
				d_stim_basal_values, d_stim_map,
				d_externalCurrent_,
				(int)nCompt_);
	}else if(UPDATE_MATRIX_APPROACH == UPDATE_MATRIX_SPMV_APPROACH){
		// Using Cusparse
		const double alpha = 1.0;
		const double beta = 0.0;

		// SPMV approach for update matrix
		cusparseDcsrmv(cusparse_handle,  CUSPARSE_OPERATION_NON_TRANSPOSE,
			nCompt_, nCompt_, num_channels, &alpha, cusparse_descr,
			d_chan_Gk, d_chan_rowPtr, d_chan_colIndex,
			d_chan_x , &beta, d_comp_Gksum);

		cusparseDcsrmv(cusparse_handle,  CUSPARSE_OPERATION_NON_TRANSPOSE,
			nCompt_, nCompt_, num_channels, &alpha, cusparse_descr,
			d_chan_GkEk, d_chan_rowPtr, d_chan_colIndex,
			d_chan_x , &beta, d_comp_GkEksum);

		update_matrix_kernel<<<BLOCKS, THREADS_PER_BLOCK>>>(d_V,
						d_HS_,
						d_comp_Gksum,
						d_comp_GkEksum,
						d_compartment_,
						d_inject_,
						d_externalCurrent_,
						(int)nCompt_);
	}else{
		// Future approaches, if any.
	}

	cudaMemcpy(perv_dynamic, d_perv_dynamic, 2*nCompt_*sizeof(double), cudaMemcpyDeviceToHost );

#ifdef PIN_POINT_ERROR
	cudaCheckError(); // Checking for cuda related errors.
#endif

}



/*
 * EXPERIMENTAL. Update matrix where data structure is in CSR format.
 */

void HSolveActive::update_csrmatrix_cuda_wrapper(){
	// ---------------------------- GKSum & GkEkSum ---------------------------------------
	int num_channels = channel_.size();
	// Using Cusparse
	const double alpha_ = 1.0;
	const double beta_ = 0.0;

	cusparseDcsrmv(cusparse_handle,  CUSPARSE_OPERATION_NON_TRANSPOSE,
		nCompt_, nCompt_, num_channels, &alpha_, cusparse_descr,
		d_chan_Gk, d_chan_rowPtr, d_chan_colIndex,
		d_chan_x , &beta_, d_comp_Gksum);

	cusparseDcsrmv(cusparse_handle,  CUSPARSE_OPERATION_NON_TRANSPOSE,
		nCompt_, nCompt_, num_channels, &alpha_, cusparse_descr,
		d_chan_GkEk, d_chan_rowPtr, d_chan_colIndex,
		d_chan_x , &beta_, d_comp_GkEksum);


	// ----------------------------- UPDATE MATRIX ----------------------------------------
	int BLOCKS = nCompt_/THREADS_PER_BLOCK;
	BLOCKS = (nCompt_%THREADS_PER_BLOCK == 0)?BLOCKS:BLOCKS+1; // Adding 1 to handle last threads

	cudaMemcpy(d_inject_, &inject_[0], nCompt_*sizeof(InjectStruct), cudaMemcpyHostToDevice);
	cudaMemcpy(d_externalCurrent_, &(externalCurrent_.front()), 2 * nCompt_ * sizeof(double), cudaMemcpyHostToDevice);

	update_csr_matrix_kernel<<<BLOCKS, THREADS_PER_BLOCK>>>(d_V,
			d_mat_values, d_main_diag_passive, d_main_diag_map, d_b,
			d_comp_Gksum,
			d_comp_GkEksum,
			d_compartment_,
			d_inject_,
			d_externalCurrent_,
			(int)nCompt_);

	cudaMemcpy(&inject_[0], d_inject_, nCompt_*sizeof(InjectStruct), cudaMemcpyDeviceToHost );

}

void HSolveActive::advance_calcium_cuda_wrapper(){

	int num_ca_pools = caConc_.size();
	int num_catarget_channels = h_catarget_channel_indices.size();

	if(ADVANCE_CALCIUM_APPROACH == ADVANCE_CALCIUM_WPT_APPROACH){
		// WPT APPROACH
		int BLOCKS = num_ca_pools/THREADS_PER_BLOCK;
		BLOCKS = (num_ca_pools%THREADS_PER_BLOCK == 0)?BLOCKS:BLOCKS+1; // Adding 1 to handle last threads
		advance_calcium_cuda_opt<<<BLOCKS,THREADS_PER_BLOCK>>>(d_catarget_channel_indices,
						d_chan_Gk, d_chan_GkEk,
						d_Vmid,
						//d_capool_values,
						d_chan_to_comp,d_capool_rowPtr,
						//d_caConc_,
						d_ca,
						d_CaConcStruct_c_, // Dynamic array
						d_CaConcStruct_CaBasal_, d_CaConcStruct_factor1_, d_CaConcStruct_factor2_, d_CaConcStruct_ceiling_, d_CaConcStruct_floor_, // Static array
						//d_caActivation_values,
						num_ca_pools);
	//}else if(ADVANCE_CALCIUM_APPROACH == ADVANCE_CALCIUM_SPMV_APPROACH){
	}else{
		// SPMV APPROACH
		double alpha = 1;
		double beta = 0;

		int BLOCKS = num_catarget_channels/THREADS_PER_BLOCK;
		BLOCKS = (num_catarget_channels%THREADS_PER_BLOCK == 0)?BLOCKS:BLOCKS+1; // Adding 1 to handle last threads

		// Find indivudual values and use CSRMV to find caActivation Values
		advance_calcium_cuda<<<BLOCKS,THREADS_PER_BLOCK>>>(d_catarget_channel_indices,
				d_chan_Gk, d_chan_GkEk,
				d_Vmid,
				d_capool_values, d_chan_to_comp,
				num_catarget_channels);

		cusparseDcsrmv(cusparse_handle,
				CUSPARSE_OPERATION_NON_TRANSPOSE,
				num_ca_pools, num_catarget_channels, num_catarget_channels ,
				&alpha, cusparse_descr,
				d_capool_values, d_capool_rowPtr, d_capool_colIndex, d_capool_onex,
				&beta, d_caActivation_values);

		BLOCKS = num_ca_pools/THREADS_PER_BLOCK;
		BLOCKS = (num_ca_pools%THREADS_PER_BLOCK == 0)?BLOCKS:BLOCKS+1; // Adding 1 to handle last threads

		advance_calcium_conc_cuda<<<BLOCKS,THREADS_PER_BLOCK>>>(d_caConc_, d_ca, d_caActivation_values, num_ca_pools);
	}

	// Sending calcium data to host
	cudaMemcpy(&(ca_[0]), d_ca, ca_.size()*sizeof(double), cudaMemcpyDeviceToHost);

#ifdef PIN_POINT_ERROR
	cudaCheckError(); // Checking for cuda related errors.
#endif
}

void HSolveActive::calculate_V_from_Vmid_wrapper(){
	int BLOCKS = (nCompt_+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK;
	calculate_V_from_Vmid<<<BLOCKS, THREADS_PER_BLOCK>>>(d_Vmid, d_V, nCompt_);
}


/*
 * Chooses whether to use WPT(Work per thread) approach or SPMV(Sparse matrix vector multiplication)
 * approach based on their averged execution time in updateMatrix module.
 */
int HSolveActive::choose_update_matrix_approach(){
	int num_repeats = 10;
	float wpt_cum_time = 0;
	float spmv_cum_time = 0;

	// Setting up cusparse information
	cusparseHandle_t cusparseH;
	cusparseCreate(&cusparseH);

	// create and setup matrix descriptors A, B & C
	cusparseMatDescr_t cuspaseDescr;
	cusparseCreateMatDescr(&cuspaseDescr);
	cusparseSetMatType(cuspaseDescr, CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatIndexBase(cuspaseDescr, CUSPARSE_INDEX_BASE_ZERO);


	int num_channels = channel_.size();
	const double alpha = 1.0;
	const double beta = 0.0;

	for (int i = 0; i < num_repeats; ++i) {
		GpuTimer timer1, timer2;

		int BLOCKS = (nCompt_+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK;
		timer1.Start();
			wpt_kernel<<<BLOCKS,THREADS_PER_BLOCK>>>(d_chan_Gk, d_chan_GkEk , d_chan_rowPtr, d_comp_Gksum, d_comp_GkEksum, nCompt_);
			cudaDeviceSynchronize();
		timer1.Stop();

		double t1 = timer1.Elapsed();
		if(i>0)	wpt_cum_time += t1;

		timer2.Start();
			cusparseDcsrmv(cusparseH,  CUSPARSE_OPERATION_NON_TRANSPOSE,
				nCompt_, nCompt_, num_channels, &alpha, cuspaseDescr,
				d_chan_Gk, d_chan_rowPtr, d_chan_colIndex,
				d_chan_x , &beta, d_comp_Gksum);

			cusparseDcsrmv(cusparseH,  CUSPARSE_OPERATION_NON_TRANSPOSE,
				nCompt_, nCompt_, num_channels, &alpha, cuspaseDescr,
				d_chan_GkEk, d_chan_rowPtr, d_chan_colIndex,
				d_chan_x , &beta, d_comp_GkEksum);
			cudaDeviceSynchronize();
		timer2.Stop();

		double t2 = timer2.Elapsed();
		if(i>0)	spmv_cum_time += t2;
		// cout << t1 << " " << t2 << endl;
		// cout << "Cumu " <<  wpt_cum_time << " " << spmv_cum_time << endl;
	}

	if(wpt_cum_time < spmv_cum_time){
		return UPDATE_MATRIX_WPT_APPROACH;
	}else{
		return UPDATE_MATRIX_SPMV_APPROACH;
	}
}

int HSolveActive::choose_advance_calcium_approach(){
	int num_repeats = 10;
	float wpt_cum_time = 0;
	float spmv_cum_time = 0;

	// Setting up cusparse information
	cusparseHandle_t cusparseH;
	cusparseCreate(&cusparseH);

	// create and setup matrix descriptors A, B & C
	cusparseMatDescr_t cuspaseDescr;
	cusparseCreateMatDescr(&cuspaseDescr);
	cusparseSetMatType(cuspaseDescr, CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatIndexBase(cuspaseDescr, CUSPARSE_INDEX_BASE_ZERO);


	int num_ca_pools = caConc_.size();
	int num_catarget_channels = h_catarget_channel_indices.size();
	double alpha = 1;
	double beta = 0;

	// Taking backup and modifying
	double* d_ca_backup, *d_CaConcStruct_c_backup;
	CaConcStruct* d_caConc_backup;

	cudaMalloc((void**)&d_ca_backup, sizeof(double)*num_ca_pools);
	cudaMalloc((void**)&d_CaConcStruct_c_backup, sizeof(double)*num_ca_pools);
	cudaMalloc((void**)&d_caConc_backup, sizeof(CaConcStruct)*num_ca_pools);

	cudaMemcpy(d_ca_backup, d_ca, num_ca_pools*sizeof(double), cudaMemcpyDeviceToDevice);
	cudaMemcpy(d_CaConcStruct_c_backup, d_CaConcStruct_c_, num_ca_pools*sizeof(double), cudaMemcpyDeviceToDevice);
	cudaMemcpy(d_caConc_backup, d_caConc_, sizeof(CaConcStruct)*num_ca_pools, cudaMemcpyDeviceToDevice);

	for (int i = 0; i < num_repeats; ++i) {
		GpuTimer timer1, timer2;

		// WPT approach
		int BLOCKS = num_ca_pools/THREADS_PER_BLOCK;
		BLOCKS = (num_ca_pools%THREADS_PER_BLOCK == 0)?BLOCKS:BLOCKS+1; // Adding 1 to handle last threads
		timer1.Start();
			advance_calcium_cuda_opt<<<BLOCKS,THREADS_PER_BLOCK>>>(d_catarget_channel_indices,
							d_chan_Gk, d_chan_GkEk,
							d_Vmid,
							//d_capool_values,
							d_chan_to_comp,d_capool_rowPtr,
							//d_caConc_,
							d_ca,
							d_CaConcStruct_c_, // Dynamic array
							d_CaConcStruct_CaBasal_, d_CaConcStruct_factor1_, d_CaConcStruct_factor2_, d_CaConcStruct_ceiling_, d_CaConcStruct_floor_, // Static array
							//d_caActivation_values,
							num_ca_pools);
		timer1.Stop();
		cudaDeviceSynchronize();
		wpt_cum_time += timer1.Elapsed();

		// SPMV approach.
		BLOCKS = num_catarget_channels/THREADS_PER_BLOCK;
		BLOCKS = (num_catarget_channels%THREADS_PER_BLOCK == 0)?BLOCKS:BLOCKS+1; // Adding 1 to handle last threads

		timer2.Start();
			// Find indivudual values and use CSRMV to find caActivation Values
			advance_calcium_cuda<<<BLOCKS,THREADS_PER_BLOCK>>>(d_catarget_channel_indices,
					d_chan_Gk, d_chan_GkEk,
					d_Vmid,
					d_capool_values, d_chan_to_comp,
					num_catarget_channels);

			cusparseDcsrmv(cusparse_handle,
					CUSPARSE_OPERATION_NON_TRANSPOSE,
					num_ca_pools, num_catarget_channels, num_catarget_channels ,
					&alpha, cusparse_descr,
					d_capool_values, d_capool_rowPtr, d_capool_colIndex, d_capool_onex,
					&beta, d_caActivation_values);

			BLOCKS = num_ca_pools/THREADS_PER_BLOCK;
			BLOCKS = (num_ca_pools%THREADS_PER_BLOCK == 0)?BLOCKS:BLOCKS+1; // Adding 1 to handle last threads

			advance_calcium_conc_cuda<<<BLOCKS,THREADS_PER_BLOCK>>>(d_caConc_, d_ca, d_caActivation_values, num_ca_pools);
		timer2.Stop();
		cudaDeviceSynchronize();

		spmv_cum_time += timer2.Elapsed();
	}

	// Restoring glory
	cudaMemcpy(d_ca, d_ca_backup, num_ca_pools*sizeof(double), cudaMemcpyDeviceToDevice);
	cudaMemcpy(d_CaConcStruct_c_, d_CaConcStruct_c_backup, num_ca_pools*sizeof(double), cudaMemcpyDeviceToDevice);
	cudaMemcpy(d_caConc_, d_caConc_backup, sizeof(CaConcStruct)*num_ca_pools, cudaMemcpyDeviceToDevice);

	if(wpt_cum_time < spmv_cum_time){
		return ADVANCE_CALCIUM_WPT_APPROACH;
	}else{
		return ADVANCE_CALCIUM_SPMV_APPROACH;
	}

}

#endif
