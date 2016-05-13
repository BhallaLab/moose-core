/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**   copyright (C) 2003-2007 Upinder S. Bhalla, Niraj Dudani and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/
#include "header.h"
#include <queue>
#include "HSolveStruct.h"
#include "HinesMatrix.h"
#include "HSolvePassive.h"
#include "HSolveActive.h"
#include "HSolve.h"
#include "../biophysics/CompartmentBase.h"
#include "../biophysics/Compartment.h"
#include "../biophysics/CaConcBase.h"
#include "ZombieCaConc.h"

#include "CudaGlobal.h"
#include "RateLookup.h"

//#define TEST_CORRECTNESS

#ifdef USE_CUDA
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/detail/static_assert.h>
using namespace thrust;
#endif
using namespace std;
#include <sys/time.h>

typedef unsigned long long u64;

/* get microseconds (us) */
u64 getTime()
{
 struct timeval tv;

 gettimeofday(&tv, NULL);

 u64 ret = tv.tv_usec;

 ret += (tv.tv_sec * 1000 * 1000);

 return ret;
}


using namespace moose;

//~ #include "ZombieCompartment.h"
//~ #include "ZombieCaConc.h"

extern ostream& operator <<( ostream& s, const HinesMatrix& m );

const int HSolveActive::INSTANT_X = 1;
const int HSolveActive::INSTANT_Y = 2;
const int HSolveActive::INSTANT_Z = 4;

HSolveActive::HSolveActive()
{
    caAdvance_ = 1;
    
#ifdef USE_CUDA    
	current_ca_position = 0;
	is_inited_ = 0;
	for(int i = 0; i < 10; ++i)
	{
		total_time[i] = 0;
	}
	total_count = 0;

	// Initializing varibales
	step_num = 0;

	cublas_handle = 0;
	cusparse_handle = 0;
	cusparse_descr = 0;
	cusolver_handle = 0;

	num_comps_with_chans = 0;
	is_initialized = false;
#endif


    // Default lookup table size
    //~ vDiv_ = 3000;    // for voltage
    //~ caDiv_ = 3000;   // for calcium
}

//////////////////////////////////////////////////////////////////////
// Solving differential equations
//////////////////////////////////////////////////////////////////////

void HSolveActive::step( ProcPtr info )
{	
    if ( nCompt_ <= 0 )
        return;

    if ( !current_.size() )
    {
        current_.resize( channel_.size() );
    }
    
#ifdef USE_CUDA
    total_count ++;
    step_num++;
#endif
    u64 start, end;
    double advanceChannelsTime;
    double calcChanCurTime;
    double updateMatTime;
    double solverTime;
    double advCalcTime;
    double advSynchanTime;
    double sendValuesTime;
    double sendSpikesTime;
    double memoryTransferTime;

#ifdef USE_CUDA
    GpuTimer advChanTimer, calcChanTimer, umTimer, solverTimer, advCalcTimer;
    advChanTimer.Start();
    	advanceChannels( info->dt );
    advChanTimer.Stop();
    advanceChannelsTime = advChanTimer.Elapsed();

    calcChanTimer.Start();
    	calculateChannelCurrents();
	calcChanTimer.Stop();
	calcChanCurTime = calcChanTimer.Elapsed();

	umTimer.Start();
		updateMatrix();
	umTimer.Stop();
	updateMatTime = umTimer.Elapsed();

	solverTimer.Start();
		HSolvePassive::forwardEliminate();
		HSolvePassive::backwardSubstitute();
		cudaMemcpy(d_Vmid, &(VMid_[0]), nCompt_*sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(d_V, &(V_[0]), nCompt_*sizeof(double), cudaMemcpyHostToDevice);

		//hinesMatrixSolverWrapper();
	solverTimer.Stop();
	solverTime = solverTimer.Elapsed();

	advCalcTimer.Start();
		advanceCalcium();
	advCalcTimer.Stop();
	advCalcTime = advCalcTimer.Elapsed();

	start = getTime();
		advanceSynChans( info );
	end = getTime();
	advSynchanTime = (end-start)/1000.0f;

	start = getTime();
		sendValues( info );
	end = getTime();
	sendValuesTime = (end-start)/1000.0f;

	start = getTime();
		sendSpikes( info );
	end = getTime();
	sendSpikesTime = (end-start)/1000.0f;

	start = getTime();
		//transfer_memory2cpu_cuda();
	end = getTime();
	memoryTransferTime = (end-start)/1000.0f;
#else

	start = getTime();
		advanceChannels( info->dt );
	end = getTime();
	advanceChannelsTime = (end-start)/1000.0f;

	start = getTime();
		calculateChannelCurrents();
	end = getTime();
	calcChanCurTime = (end-start)/1000.0f;

	start = getTime();
		updateMatrix();
		//updateForwardFlowMatrix();
		//updatePervasiveFlowMatrix();
	end = getTime();
	updateMatTime = (end-start)/1000.0f;

	start = getTime();
		HSolvePassive::forwardEliminate();
		HSolvePassive::backwardSubstitute();

		// Using forward flow solution
		//forwardFlowSolver();

		// Using pervasive flow solution
		//pervasiveFlowSolver();
	end = getTime();
	solverTime = (end-start)/1000.0f;

	start = getTime();
		advanceCalcium();
	end = getTime();
	advCalcTime = (end-start)/1000.0f;

	start = getTime();
		advanceSynChans( info );
	end = getTime();
	advSynchanTime = (end-start)/1000.0f;

	start = getTime();
		sendValues( info );
	end = getTime();
	sendValuesTime = (end-start)/1000.0f;

	start = getTime();
		sendSpikes( info );
	end = getTime();
	sendSpikesTime = (end-start)/1000.0f;

#endif

    externalCurrent_.assign( externalCurrent_.size(), 0.0 );
    
}

void HSolveActive::calculateChannelCurrents()
{
#ifdef USE_CUDA
	calculate_channel_currents_cuda_wrapper();
	cudaSafeCall(cudaMemcpy(&current_[0], d_current_, current_.size()*sizeof(CurrentStruct), cudaMemcpyDeviceToHost));
#else
    vector< ChannelStruct >::iterator ichan;
    vector< CurrentStruct >::iterator icurrent = current_.begin();

    if ( state_.size() != 0 )
    {
        double* istate = &state_[ 0 ];

        for ( ichan = channel_.begin(); ichan != channel_.end(); ++ichan )
        {
            ichan->process( istate, *icurrent );
            ++icurrent;
        }
    }
#endif
}

void HSolveActive::updateMatrix()
{

#ifdef USE_CUDA
	// Updates HS matrix and sends it to CPU
	if ( HJ_.size() != 0 )
		memcpy( &HJ_[ 0 ], &HJCopy_[ 0 ], sizeof( double ) * HJ_.size() );
	update_matrix_cuda_wrapper();

	// Updates CSR matrix and sends it to CPU.
	//update_csrmatrix_cuda_wrapper();

#else
	// CPU PART
	/*
	 * Copy contents of HJCopy_ into HJ_. Cannot do a vector assign() because
	 * iterators to HJ_ get invalidated in MS VC++
	 * TODO Is it needed to do a memcpy each time?
	 */
	if ( HJ_.size() != 0 )
			memcpy( &HJ_[ 0 ], &HJCopy_[ 0 ], sizeof( double ) * HJ_.size() );

    double GkSum, GkEkSum; vector< CurrentStruct >::iterator icurrent = current_.begin();
    vector< currentVecIter >::iterator iboundary = currentBoundary_.begin();
    vector< double >::iterator ihs = HS_.begin();
    vector< double >::iterator iv = V_.begin();

    vector< CompartmentStruct >::iterator ic;
    for ( ic = compartment_.begin(); ic != compartment_.end(); ++ic )
    {
        GkSum   = 0.0;
        GkEkSum = 0.0;
        for ( ; icurrent < *iboundary; ++icurrent )
        {
            GkSum   += icurrent->Gk;
            GkEkSum += icurrent->Gk * icurrent->Ek;
        }

        *ihs = *( 2 + ihs ) + GkSum;
        *( 3 + ihs ) = *iv * ic->CmByDt + ic->EmByRm + GkEkSum;

        ++iboundary, ihs += 4, ++iv;
    }
#ifdef USE_CUDA
    for(int i=0;i<inject_.size();i++){
    	HS_[ 4 * i + 3 ] += inject_[i].injectVarying + inject_[i].injectBasal;
    	inject_[i].injectVarying = 0;
    }
#else
    map< unsigned int, InjectStruct >::iterator inject;
    for ( inject = inject_.begin(); inject != inject_.end(); ++inject )
    {
        unsigned int ic = inject->first;
        InjectStruct& value = inject->second;

        HS_[ 4 * ic + 3 ] += value.injectVarying + value.injectBasal;

        value.injectVarying = 0.0;
    }
#endif
    // Synapses are being handled as external channels.
    //~ double Gk, Ek;
    //~ vector< SynChanStruct >::iterator isyn;
    //~ for ( isyn = synchan_.begin(); isyn != synchan_.end(); ++isyn ) {
    //~ get< double >( isyn->elm_, synGkFinfo, Gk );
    //~ get< double >( isyn->elm_, synEkFinfo, Ek );
    //~
    //~ unsigned int ic = isyn->compt_;
    //~ HS_[ 4 * ic ] += Gk;
    //~ HS_[ 4 * ic + 3 ] += Gk * Ek;
    //~ }

    ihs = HS_.begin();
    vector< double >::iterator iec;
    for ( iec = externalCurrent_.begin(); iec != externalCurrent_.end(); iec += 2 )
    {
        *ihs += *iec;
        *( 3 + ihs ) += *( iec + 1 );

        ihs += 4;
    }
#endif
    stage_ = 0;    // Update done.
}
void HSolveActive::updateForwardFlowMatrix()
{
	double GkSum, GkEkSum; vector< CurrentStruct >::iterator icurrent = current_.begin();
	vector< currentVecIter >::iterator iboundary = currentBoundary_.begin();
	for (unsigned int i = 0; i < compartment_.size(); ++i)
	{
		GkSum   = 0.0;
		GkEkSum = 0.0;
		for ( ; icurrent < *iboundary; ++icurrent )
		{
			GkSum   += icurrent->Gk;
			GkEkSum += icurrent->Gk * icurrent->Ek;
		}

		ff_system[nCompt_+i] = ff_system[2*nCompt_+i] + GkSum;
		ff_system[3*nCompt_+i] = V_[i] * compartment_[i].CmByDt + compartment_[i].EmByRm + GkEkSum;

		++iboundary;
	}

	#ifdef USE_CUDA
		for(unsigned int i=0;i<inject_.size();i++){
			ff_system[ 3*nCompt_ + i ] += inject_[i].injectVarying + inject_[i].injectBasal;
			inject_[i].injectVarying = 0;
		}
	#else
		map< unsigned int, InjectStruct >::iterator inject;
		for ( inject = inject_.begin(); inject != inject_.end(); ++inject )
		{
			unsigned int ic = inject->first;
			InjectStruct& value = inject->second;

			ff_system[3*nCompt_+ic] += value.injectVarying + value.injectBasal;
			value.injectVarying = 0.0;
		}
	#endif

    vector< double >::iterator iec;
    for (unsigned int i = 0; i < nCompt_; i++)
    {
    	ff_system[nCompt_+i] += externalCurrent_[2*i];
    	ff_system[3*nCompt_+i] += externalCurrent_[2*i+1];
    }

    stage_ = 0;

}

void HSolveActive::forwardFlowSolver(){
	/*
	for (int i = 0; i < V_.size(); ++i) {
		if(i==0) cout << "Voltages" << endl;
		cout << V_[i] << endl;
	}
	*/

	//print_tridiagonal_matrix_system(ff_system, ff_offdiag_mapping, nCompt_);

	// Forward Elimination
	int parentId;
	for(unsigned int i=1;i<nCompt_;i++){
		parentId = ff_offdiag_mapping[i-1];
		ff_system[nCompt_+parentId] -= (ff_system[i])*(ff_system[i])/ff_system[nCompt_+i-1];
		ff_system[3*nCompt_+parentId] -= (ff_system[3*nCompt_+(i-1)]*ff_system[i])/ff_system[nCompt_+i-1];
	}

	// Backward Substitution
	VMid_[nCompt_-1] = ff_system[3*nCompt_ + (nCompt_-1)]/ff_system[2*nCompt_-1];
	V_[nCompt_-1] = 2*VMid_[nCompt_-1] - V_[nCompt_-1];
	int columnId;
	for(int i=nCompt_-2;i>=0;i--){
		VMid_[i] = (ff_system[3*nCompt_+i]-VMid_[ff_offdiag_mapping[i]]*ff_system[i+1])/ff_system[nCompt_+i];
		V_[i] = 2*VMid_[i] - V_[i];
	}

	stage_ = 2;
}

void HSolveActive::updatePervasiveFlowMatrix(){

	// Copying initial matrix
	memcpy(upper_mat.values, upper_mat_values_copy, upper_mat.nnz*sizeof(double));
	memcpy(lower_mat.values, lower_mat_values_copy, lower_mat.nnz*sizeof(double));

	double GkSum, GkEkSum; vector< CurrentStruct >::iterator icurrent = current_.begin();
	vector< currentVecIter >::iterator iboundary = currentBoundary_.begin();
	for (unsigned int i = 0; i < compartment_.size(); ++i)
	{
		GkSum   = 0.0;
		GkEkSum = 0.0;
		for ( ; icurrent < *iboundary; ++icurrent )
		{
			GkSum   += icurrent->Gk;
			GkEkSum += icurrent->Gk * icurrent->Ek;
		}

		upper_mat.values[per_mainDiag_map[i]] = per_mainDiag_passive[i] + GkSum;
		per_rhs[i] = V_[i] * compartment_[i].CmByDt + compartment_[i].EmByRm + GkEkSum;

		++iboundary;
	}

	#ifdef USE_CUDA
		for(unsigned int i=0;i<inject_.size();i++){
			per_rhs[i] += inject_[i].injectVarying + inject_[i].injectBasal;
			inject_[i].injectVarying = 0;
		}
	#else
	    map< unsigned int, InjectStruct >::iterator inject;
	    for ( inject = inject_.begin(); inject != inject_.end(); ++inject )
	    {
	        unsigned int ic = inject->first;
	        InjectStruct& value = inject->second;

	        per_rhs[ic] += value.injectVarying + value.injectBasal;
	        value.injectVarying = 0.0;
	    }
	#endif


    vector< double >::iterator iec;
    for (unsigned int i = 0; i < nCompt_; i++)
    {
    	upper_mat.values[per_mainDiag_map[i]] += externalCurrent_[2*i];
    	per_rhs[i] += externalCurrent_[2*i+1];
    }

    stage_ = 0;
}

void HSolveActive::pervasiveFlowSolver(){
   	// TODO
	// Gauss elimination
	for(int i=0;i<lower_mat.nnz;i++){
		double scaling = lower_mat.values[i]/upper_mat.values[upper_mat.rowPtr[lower_mat.colIndex[i]]];
		// UT-LT
		for(int j=ut_lt_rowPtr[i];j < ut_lt_rowPtr[i+1]; j++){
			lower_mat.values[ut_lt_lower[j]] -= (upper_mat.values[ut_lt_upper[j]]*scaling);
		}

		// UT-UT
		for(int j=ut_ut_rowPtr[i]; j < ut_ut_rowPtr[i+1]; j++){
			upper_mat.values[ut_ut_lower[j]] -= (upper_mat.values[ut_ut_upper[j]]*scaling);
		}

		// RHS
		per_rhs[lower_mat.rowIndex[i]] -= (per_rhs[lower_mat.colIndex[i]]*scaling);
	}

	//print_csr_matrix(upper_mat);
	//print_matrix(rhs,num_comp,1);

	// Backward substitution
	for(int i=nCompt_-1;i>=0;i--){
		double sum = 0;
		for(int j=upper_mat.rowPtr[i]+1;j<upper_mat.rowPtr[i+1];j++){
			sum += (upper_mat.values[j]*VMid_[upper_mat.colIndex[j]]);
		}
		VMid_[i] = (per_rhs[i]-sum)/upper_mat.values[upper_mat.rowPtr[i]];
	}

	for (unsigned int i = 0; i < nCompt_; ++i) {
		V_[i] = 2*VMid_[i] - V_[i];
	}

	stage_ = 2;
}

void HSolveActive::advanceCalcium()
{
#ifdef USE_CUDA
	/*
	 * Disabling it as of now as CPU is faster.
	cudaMemset(d_caActivation_values, 0, ca_.size()*sizeof(double));

	advance_calcium_cuda_wrapper();

	cudaMemcpy(&(ca_[0]), d_ca, ca_.size()*sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(&(caConc_[0]), d_caConc_, caConc_.size()*sizeof(CaConcStruct), cudaMemcpyDeviceToHost);
	*/
#endif //#else

    vector< double* >::iterator icatarget = caTarget_.begin();
    vector< double >::iterator ivmid = VMid_.begin();
    vector< CurrentStruct >::iterator icurrent = current_.begin();
    vector< currentVecIter >::iterator iboundary = currentBoundary_.begin();

    /*
     * caAdvance_: This flag determines how current flowing into a calcium pool
     * is computed. A value of 0 means that the membrane potential at the
     * beginning of the time-step is used for the calculation. This is how
     * GENESIS does its computations. A value of 1 means the membrane potential
     * at the middle of the time-step is used. This is the correct way of
     * integration, and is the default way.
     */
    if ( caAdvance_ == 1 )
    {
        for ( ; iboundary != currentBoundary_.end(); ++iboundary )
        {
            for ( ; icurrent < *iboundary; ++icurrent )
            {
                if ( *icatarget )
                    **icatarget += icurrent->Gk * ( icurrent->Ek - *ivmid );

                ++icatarget;
            }

            ++ivmid;
        }
    }
    else if ( caAdvance_ == 0 )
    {
        vector< double >::iterator iv = V_.begin();
        double v0;

        for ( ; iboundary != currentBoundary_.end(); ++iboundary )
        {
            for ( ; icurrent < *iboundary; ++icurrent )
            {
                if ( *icatarget )
                {
                    v0 = ( 2 * *ivmid - *iv );

                    **icatarget += icurrent->Gk * ( icurrent->Ek - v0 );
                }

                ++icatarget;
            }

            ++ivmid, ++iv;
        }
    }

    vector< CaConcStruct >::iterator icaconc;
    vector< double >::iterator icaactivation = caActivation_.begin();
    vector< double >::iterator ica = ca_.begin();
    for ( icaconc = caConc_.begin(); icaconc != caConc_.end(); ++icaconc )
    {
        *ica = icaconc->process( *icaactivation );
        ++ica, ++icaactivation;
    }

    caActivation_.assign( caActivation_.size(), 0.0 );
//#endif
}

void HSolveActive::advanceChannels( double dt )
{

#ifdef USE_CUDA

    // Useful variables
    int num_comps = V_.size();
    int num_gates = channel_.size()*3;

	if(!is_initialized){
		// TODO Move it to appropriate place
		double* test_gate_values = new double[num_gates]();
		double pivot_thresh = 0;

		// Initializing device Vm and Ca pools
		cudaSafeCall(cudaMemcpy(d_V, &(V_.front()), nCompt_ * sizeof(double), cudaMemcpyHostToDevice));
		cudaSafeCall(cudaMemcpy(d_ca, &(ca_.front()), ca_.size()*sizeof(double), cudaMemcpyHostToDevice));

		// Initializing device inject and external current.
		cudaMemcpy(d_inject_, &inject_[0], nCompt_*sizeof(InjectStruct), cudaMemcpyHostToDevice);
		cudaMemcpy(d_externalCurrent_, &(externalCurrent_.front()), 2 * nCompt_ * sizeof(double), cudaMemcpyHostToDevice);

		// Initializing device gate fraction values.
		for(unsigned int i=0;i<h_gate_expand_indices.size();i++){
			test_gate_values[h_gate_expand_indices[i]] = state_[i];
		}
		cudaMemcpy(d_gate_values, test_gate_values, num_gates*sizeof(double), cudaMemcpyHostToDevice);

		// Setting up cusparse information
		cusparseCreate(&cusparse_handle);
		cublasCreate_v2(&cublas_handle);
		cusolverSpCreate(&cusolver_handle);

		// create and setup matrix descriptors A, B & C
		cusparseCreateMatDescr(&cusparse_descr);
		cusparseSetMatType(cusparse_descr, CUSPARSE_MATRIX_TYPE_GENERAL);
		cusparseSetMatIndexBase(cusparse_descr, CUSPARSE_INDEX_BASE_ZERO);

		cusolverStatus_t cusolver_status;

		// Analyzing the matrix structure
	 	cusolver_status =  cusolverSpCreateCsrluInfoHost(&infoA); // Creating info of A
		cusolver_status = cusolverSpXcsrluAnalysisHost(cusolver_handle,
														nCompt_,
														mat_nnz,
														cusparse_descr,
														h_mat_rowPtr, h_mat_colIndex,
														infoA);

		// Getting the memory requirements for LU factorization.

		cusolver_status = cusolverSpDcsrluBufferInfoHost(cusolver_handle,
														nCompt_,
														mat_nnz,
														cusparse_descr,
														h_mat_values, h_mat_rowPtr, h_mat_colIndex,
														infoA, &internalDataInBytes, &workspaceInBytes);

		// Allocate memory for CPU solver.
		internalBuffer = (double*) malloc(internalDataInBytes);
		workspaceBuffer = (double*) malloc(workspaceInBytes);

		is_initialized = true;
	}

	// Calling the kernels
	cudaSafeCall(cudaMemcpy(d_V, &(V_.front()), nCompt_ * sizeof(double), cudaMemcpyHostToDevice));
	cudaSafeCall(cudaMemcpy(d_ca, &(ca_.front()), ca_.size()*sizeof(double), cudaMemcpyHostToDevice));

	get_lookup_rows_and_fractions_cuda_wrapper(dt); // Gets lookup values for Vm and Ca_.
	advance_channels_cuda_wrapper(dt); // Advancing fraction values.
	get_compressed_gate_values_wrapper(); // Getting values of new state

	cudaSafeCall(cudaMemcpy(&state_[0], d_state_, state_.size()*sizeof(double), cudaMemcpyDeviceToHost));

#else

    vector< double >::iterator iv;
    vector< double >::iterator istate = state_.begin();
    vector< int >::iterator ichannelcount = channelCount_.begin();
    vector< ChannelStruct >::iterator ichan = channel_.begin();
    vector< ChannelStruct >::iterator chanBoundary;
    vector< unsigned int >::iterator icacount = caCount_.begin();
    vector< double >::iterator ica = ca_.begin();
    vector< double >::iterator caBoundary;
    vector< LookupColumn >::iterator icolumn = column_.begin();
    vector< LookupRow >::iterator icarowcompt;
    vector< LookupRow* >::iterator icarow = caRow_.begin();

    LookupRow vRow;

	u64 cpu_advchan_start = getTime();

    double C1, C2;

    for ( iv = V_.begin(); iv != V_.end(); ++iv )
    {
        vTable_.row( *iv, vRow );
        icarowcompt = caRowCompt_.begin();
        caBoundary = ica + *icacount;
        
        for ( ; ica < caBoundary; ++ica )
        {
            caTable_.row( *ica, * icarowcompt );
            ++icarowcompt;
        }   
        /*
         * Optimize by moving "if ( instant )" outside the loop, because it is
         * rarely used. May also be able to avoid "if ( power )".
         *
         * Or not: excellent branch predictors these days.
         *
         * Will be nice to test these optimizations.
         */
        chanBoundary = ichan + *ichannelcount;
        for ( ; ichan < chanBoundary; ++ichan )
        {
            if ( ichan->Xpower_ > 0.0 )
            {
                vTable_.lookup( *icolumn, vRow, C1, C2 );
                //~ *istate = *istate * C1 + C2;
                //~ *istate = ( C1 + ( 2 - C2 ) * *istate ) / C2;
                if ( ichan->instant_ & INSTANT_X )
                    *istate = C1 / C2;
                else
                {
                    double temp = 1.0 + dt / 2.0 * C2;
                    *istate = ( *istate * ( 2.0 - temp ) + dt * C1 ) / temp;
                }

                ++icolumn, ++istate;
            }

            if ( ichan->Ypower_ > 0.0 )
            {
                vTable_.lookup( *icolumn, vRow, C1, C2 );
                //~ *istate = *istate * C1 + C2;
                //~ *istate = ( C1 + ( 2 - C2 ) * *istate ) / C2;
                if ( ichan->instant_ & INSTANT_Y )
                    *istate = C1 / C2;
                else
                {
                    double temp = 1.0 + dt / 2.0 * C2;
                    *istate = ( *istate * ( 2.0 - temp ) + dt * C1 ) / temp;
                }

                ++icolumn, ++istate;
            }

            if ( ichan->Zpower_ > 0.0 )
            {
                LookupRow* caRow = *icarow;
                if ( caRow )
                {
                    caTable_.lookup( *icolumn, *caRow, C1, C2 );
                }
                else
                {
                    vTable_.lookup( *icolumn, vRow, C1, C2 );
                }

                //~ *istate = *istate * C1 + C2;
                //~ *istate = ( C1 + ( 2 - C2 ) * *istate ) / C2;
                if ( ichan->instant_ & INSTANT_Z )
                    *istate = C1 / C2;
                else
                {
                    double temp = 1.0 + dt / 2.0 * C2;
                    *istate = ( *istate * ( 2.0 - temp ) + dt * C1 ) / temp;
                }

                ++icolumn, ++istate, ++icarow;
            }
        }

        ++ichannelcount, ++icacount;
    }

    u64 cpu_advchan_end = getTime();

#endif
      
}

#ifdef USE_CUDA

/* Used to allocate device memory on GPU for Hsolve variables */
void HSolveActive::allocate_hsolve_memory_cuda(){

	// Important numbers
	int num_compts = V_.size();
	int num_channels = channel_.size() ;
	int num_gates = 3*num_channels;
	int num_cmprsd_gates = state_.size();
	int num_Ca_pools = ca_.size();

	// LookUp Tables
	int V_table_size = vTable_.get_table().size();
	int Ca_table_size = caTable_.get_table().size();

	cudaMalloc((void **)&(d_V), num_compts * sizeof(double));
	cudaMalloc((void **)&(d_V_table), V_table_size * sizeof(double));

	cudaMalloc((void **)&(d_ca), num_Ca_pools * sizeof(double));
	cudaMalloc((void **)&(d_Ca_table), Ca_table_size * sizeof(double));

	cudaMalloc((void**)&d_gate_values, num_gates*sizeof(double));
	cudaMalloc((void**)&d_gate_powers, num_gates*sizeof(double));
	cudaMalloc((void**)&d_gate_columns, num_gates*sizeof(double));
	cudaMalloc((void**)&d_gate_ca_index, num_gates*sizeof(int));

	cudaMalloc((void**)&d_state_, num_cmprsd_gates*sizeof(double));
	cudaMalloc((void**)&d_gate_expand_indices, num_cmprsd_gates*sizeof(int));

	// Channel related

	cudaMalloc((void**)&d_chan_instant, num_channels*sizeof(int));
	cudaMalloc((void**)&d_chan_modulation, num_channels*sizeof(double));
	cudaMalloc((void**)&d_chan_Gbar, num_channels*sizeof(double));
	cudaMalloc((void**)&d_chan_to_comp, num_channels*sizeof(double));
	cudaMalloc((void**)&d_chan_Gk, num_channels*sizeof(double));
	cudaMalloc((void**)&d_chan_GkEk, num_channels*sizeof(double));

	cudaMalloc((void**)&d_comp_Gksum, num_compts*sizeof(double));
	cudaMemset(d_comp_Gksum, 0, num_compts*sizeof(double));
	cudaMalloc((void**)&d_comp_GkEksum, num_compts*sizeof(double));
	cudaMemset(d_comp_GkEksum, 0, num_compts*sizeof(double));
	cudaMalloc((void**)&d_externalCurrent_, 2*num_compts*sizeof(double));


	cudaMalloc((void**)&d_current_, current_.size()*sizeof(CurrentStruct));
	cudaMalloc((void**)&d_inject_, inject_.size()*sizeof(InjectStruct));
	cudaMalloc((void**)&d_compartment_, compartment_.size()*sizeof(CompartmentStruct));
	cudaMalloc((void**)&d_caConc_, caConc_.size()*sizeof(CaConcStruct));

	// Compartment related

	// Hines Matrix related
	cudaMalloc((void**)&d_HS_, HS_.size()*sizeof(double));

	cudaMalloc((void**)&d_chan_colIndex, num_channels*sizeof(int));
	cudaMalloc((void**)&d_chan_rowPtr, (nCompt_+1)*sizeof(int));
	cudaMalloc((void**)&d_chan_x, nCompt_*sizeof(double));

	// Conjugate Gradient related
	cudaMalloc((void**)&d_Vmid, nCompt_*sizeof(double));
	cudaMalloc((void**)&d_p, nCompt_*sizeof(double));
	cudaMalloc((void**)&d_Ax, nCompt_*sizeof(double));
	cudaMalloc((void**)&d_r, nCompt_*sizeof(double));
	cudaMalloc((void**)&d_x, nCompt_*sizeof(double));

	// Intermediate data for computation
	cudaMalloc((void**)&d_V_rows, num_compts*sizeof(int));
	cudaMalloc((void**)&d_V_fractions, num_compts*sizeof(double));

	cudaMalloc((void**)&d_Ca_rows, num_Ca_pools*sizeof(int));
	cudaMalloc((void**)&d_Ca_fractions, num_Ca_pools*sizeof(double));

	cudaMalloc((void**)&d_temp_keys, num_compts*sizeof(int));
	cudaMalloc((void**)&d_temp_values, num_compts*sizeof(double));

	// AdvanceCalcium related
	cudaMalloc((void**)&d_capool_values, h_catarget_channel_indices.size()*sizeof(double));
	cudaMalloc((void**)&d_capool_onex, h_catarget_channel_indices.size()*sizeof(double));

}

void HSolveActive::copy_table_data_cuda(){
	// Transfer lookup table data on to GPU.
	vector<double> V_table_data = vTable_.get_table();
	vector<double> Ca_table_data = caTable_.get_table();

	cudaMemcpy(d_V_table, &(V_table_data.front()),
									V_table_data.size() * sizeof(double),
									cudaMemcpyHostToDevice);
	cudaMemcpy(d_Ca_table, &(Ca_table_data.front()),
										Ca_table_data.size() * sizeof(double),
										cudaMemcpyHostToDevice);
}

void HSolveActive::copy_hsolve_information_cuda(){
	int num_compts = V_.size();
	int num_channels = channel_.size();
	int num_gates = 3*num_channels;
	int num_cmprsd_gates = state_.size();
	int num_Ca_pools = ca_.size();

	// Gate variables
	double h_gate_values[num_gates];
	double h_gate_powers[num_gates];
	unsigned int h_gate_columns[num_gates];
	int h_gate_ca_index[num_gates];

	// Channel variables
	double h_chan_Gbar[num_channels];
	int h_chan_instant[num_channels];
	double h_chan_modulation[num_channels];
	unsigned int h_chan_to_comp[num_channels];

	// Transferring Vm and Ca concentration values to GPU
	cudaMemcpy(d_V, &(V_.front()), num_compts*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_ca, &(ca_.front()), num_Ca_pools*sizeof(double), cudaMemcpyHostToDevice);


	// Gathering data for each channel
	for(int i=0;i<num_channels;i++){
		h_chan_Gbar[i] = channel_[i].Gbar_;

		h_chan_instant[i] = channel_[i].instant_;
		h_chan_modulation[i] = channel_[i].modulation_;

		// Channel to Compartment Info
		h_chan_to_comp[i] = chan2compt_[i];
	}

	// Transferring channel data to GPU
	cudaMemcpy(d_chan_Gbar, h_chan_Gbar, num_channels * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_chan_instant, h_chan_instant, num_channels * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_chan_modulation, h_chan_modulation, num_channels * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_chan_to_comp, h_chan_to_comp, num_channels * sizeof(unsigned int), cudaMemcpyHostToDevice);

	// Constructing ca and channel row ptrs with nCompt as rows.
	int ca_rowPtr[V_.size()+1];
	int chan_rowPtr[V_.size()+1];
	int sum1 = 0, sum2 = 0;
	for(unsigned int i=0;i<=V_.size();i++){
		ca_rowPtr[i] = sum1;
		chan_rowPtr[i] = sum2;

		if(i < V_.size()){
			// Last one should be just set.
			sum1 += caCount_[i];
			sum2 += channelCount_[i];
		}
	}

	// Gathering gate information and separating gates (with power < 0) , (with vm dependent) , (with ca dependent)
	int cmprsd_gate_index = 0; // If the logic is true cmprsd_gate_index value at the end of for loop = # of gates with powers > 0
	for(unsigned int i=0;i<V_.size();i++){
		for(int j=chan_rowPtr[i]; j<chan_rowPtr[i+1]; j++){

			// Setting powers
			h_gate_powers[3*j] = channel_[j].Xpower_;
			h_gate_powers[3*j+1] = channel_[j].Ypower_;
			h_gate_powers[3*j+2] = channel_[j].Zpower_;

			for(int k=0;k<3;k++){
				h_gate_ca_index[3*j+k] = -1;
				if(h_gate_powers[3*j+k] > 0){
					// Setting column index and values
					h_gate_columns[3*j+k] = column_[cmprsd_gate_index].column;
					h_gate_values[3*j+k] = state_[cmprsd_gate_index];
					cmprsd_gate_index++; // cmprsd_gate_index is incremented only if power > 0 is found.

					// Partitioning of vm and ca dependent gates.
					if(k == 2 && caDependIndex_[j] != -1){
						h_gate_ca_index[3*j+k] = ca_rowPtr[i] + caDependIndex_[j];
						h_cagate_expand_indices.push_back(3*j+k);
						h_cagate_capool_indices.push_back(ca_rowPtr[i] + caDependIndex_[j]);
					}else{
						h_vgate_expand_indices.push_back(3*j+k);
						h_vgate_compt_indices.push_back((int)chan2compt_[j]);
					}
				}else{
					h_gate_columns[3*j+k] = 0;
					h_gate_values[3*j+k] = 0;
				}
			}
		}
	}
	assert(cmprsd_gate_index == num_cmprsd_gates);


	// Allocating memory
	cudaMalloc((void**)&d_vgate_expand_indices, h_vgate_expand_indices.size()* sizeof(int));
	cudaMalloc((void**)&d_vgate_compt_indices, h_vgate_compt_indices.size()* sizeof(int));

	cudaMalloc((void**)&d_cagate_expand_indices, h_cagate_expand_indices.size()* sizeof(int));
	cudaMalloc((void**)&d_cagate_capool_indices, h_cagate_capool_indices.size()* sizeof(int));

	cudaMalloc((void**)&d_catarget_channel_indices, h_catarget_channel_indices.size()* sizeof(int));
	cudaMalloc((void**)&d_catarget_capool_indices, h_catarget_capool_indices.size()* sizeof(int));
	cudaMalloc((void**)&d_caActivation_values, ca_.size()* sizeof(double));

	cudaMalloc((void**)&d_capool_rowPtr, (num_Ca_pools+1)*sizeof(int));
	cudaMalloc((void**)&d_capool_colIndex, h_catarget_channel_indices.size()*sizeof(int));

	cudaMemcpy(d_gate_values, h_gate_values, num_gates * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_gate_powers, h_gate_powers, num_gates * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_gate_columns, h_gate_columns, num_gates * sizeof(int), cudaMemcpyHostToDevice);

	cudaMemcpy(d_gate_ca_index, h_gate_ca_index, num_gates * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_gate_expand_indices, &(h_gate_expand_indices.front()), h_gate_expand_indices.size()* sizeof(int), cudaMemcpyHostToDevice);

	cudaMemcpy(d_vgate_expand_indices, &(h_vgate_expand_indices.front()), h_vgate_expand_indices.size()* sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_vgate_compt_indices, &(h_vgate_compt_indices.front()), h_vgate_compt_indices.size()* sizeof(int), cudaMemcpyHostToDevice);

	cudaMemcpy(d_cagate_expand_indices, &(h_cagate_expand_indices.front()), h_cagate_expand_indices.size()* sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_cagate_capool_indices, &(h_cagate_capool_indices.front()), h_cagate_capool_indices.size()* sizeof(int), cudaMemcpyHostToDevice);

	cudaMemcpy(d_catarget_channel_indices, &(h_catarget_channel_indices.front()), h_catarget_channel_indices.size()* sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_catarget_capool_indices, &(h_catarget_capool_indices.front()), h_catarget_capool_indices.size()* sizeof(int), cudaMemcpyHostToDevice);


	// Current Data
	cudaMemcpy(d_externalCurrent_, &(externalCurrent_.front()), 2 * num_compts * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_current_, &(current_.front()), current_.size()*sizeof(CurrentStruct), cudaMemcpyHostToDevice);
	cudaMemcpy(d_inject_, &(inject_.front()), inject_.size()*sizeof(InjectStruct), cudaMemcpyHostToDevice);
	cudaMemcpy(d_compartment_, &(compartment_.front()), compartment_.size()*sizeof(CompartmentStruct), cudaMemcpyHostToDevice);
	cudaMemcpy(d_caConc_, &(caConc_.front()), caConc_.size()*sizeof(CaConcStruct), cudaMemcpyHostToDevice);

	// Hines Matrix related
	cudaMemcpy(d_HS_, &(HS_.front()), HS_.size()*sizeof(double), cudaMemcpyHostToDevice);

	int* chan_colIndex = new int[num_channels]();
	double* chan_x = new double[nCompt_];

	std::fill_n(chan_x, nCompt_, 1);

	// Filling column indices
	for(unsigned int i=0;i<nCompt_;i++){
		for(int j=chan_rowPtr[i];j<chan_rowPtr[i+1];j++){
			chan_colIndex[j] = j-chan_rowPtr[i];
		}
	}

	cudaMemcpy(d_chan_x, chan_x, nCompt_*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_chan_colIndex, chan_colIndex , num_channels*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_chan_rowPtr, chan_rowPtr, (nCompt_+1)*sizeof(int), cudaMemcpyHostToDevice);

	int num_unique_keys = 0;
	set<int> keys;
	for(unsigned int i=0;i<chan2compt_.size();i++){
		keys.insert(chan2compt_[i]);
	}
	num_unique_keys = keys.size();

	num_comps_with_chans = num_unique_keys; // Contains number of compartments with >= 1 chans

	// Ca Target related
	int num_catarget_chans = h_catarget_channel_indices.size();
	int* temp_rowPtr = new int[num_Ca_pools+1]();
	int* temp_colIndex = new int[num_catarget_chans]();

	// Setting up row count
	for(unsigned int i=0;i<h_catarget_capool_indices.size();i++){
		temp_rowPtr[h_catarget_capool_indices[i]]++;
	}

	// Setting up row pointer
	int sum = 0, temp;
	for(int i=0;i<num_Ca_pools+1;i++){
		temp = temp_rowPtr[i];
		temp_rowPtr[i] = sum;
		sum += temp;
	}

	// Setting up column indices
	for(int i=0;i<num_Ca_pools;i++){
		for(int j=temp_rowPtr[i];j<temp_rowPtr[i+1];j++){
			temp_colIndex[j] = j-temp_rowPtr[i];
		}
	}

	cudaMemcpy(d_capool_rowPtr, temp_rowPtr, (num_Ca_pools+1)*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_capool_colIndex, temp_colIndex, num_catarget_chans*sizeof(int), cudaMemcpyHostToDevice);

	double temp_x[num_catarget_chans];
	for(int i=0;i<num_catarget_chans;i++) temp_x[i] = 1;
	cudaMemcpy(d_capool_onex, temp_x, num_catarget_chans*sizeof(double), cudaMemcpyHostToDevice);

}

void HSolveActive::transfer_memory2cpu_cuda(){
	cudaSafeCall(cudaMemcpy(&state_[0], d_state_, state_.size()*sizeof(double), cudaMemcpyDeviceToHost));
	cudaSafeCall(cudaMemcpy(&current_[0], d_current_, current_.size()*sizeof(CurrentStruct), cudaMemcpyDeviceToHost));
	cudaSafeCall(cudaMemcpy(&(V_[0]), d_V, nCompt_*sizeof(double), cudaMemcpyDeviceToHost));
	cudaSafeCall(cudaMemcpy(&(ca_[0]), d_ca, ca_.size()*sizeof(double), cudaMemcpyDeviceToHost));
	cudaSafeCall(cudaMemcpy(&(caConc_[0]), d_caConc_, caConc_.size()*sizeof(CaConcStruct), cudaMemcpyDeviceToHost));
}

LookupColumn * HSolveActive::get_column_d()
{
	return column_d;
}
#endif

/**
 * SynChans are currently not under solver's control
 */
void HSolveActive::advanceSynChans( ProcPtr info )
{
    return;
}

void HSolveActive::sendSpikes( ProcPtr info )
{
    vector< SpikeGenStruct >::iterator ispike;
    for ( ispike = spikegen_.begin(); ispike != spikegen_.end(); ++ispike )
        ispike->send( info );
}

/**
 * This function dispatches state values via any source messages on biophysical
 * objects which have been taken over.
 *
 */
void HSolveActive::sendValues( ProcPtr info )
{
    vector< unsigned int >::iterator i;

    for ( i = outVm_.begin(); i != outVm_.end(); ++i )
        moose::Compartment::VmOut()->send(
            //~ ZombieCompartment::VmOut()->send(
            compartmentId_[ *i ].eref(),
            V_[ *i ]
        );

    for ( i = outCa_.begin(); i != outCa_.end(); ++i )
        //~ CaConc::concOut()->send(
        CaConcBase::concOut()->send(
            caConcId_[ *i ].eref(),
            ca_[ *i ]
        );
}
