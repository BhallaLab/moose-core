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
#include "Gpu_timer.h"
#endif
using namespace std;
#include <sys/time.h>


#include <fstream>

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

	// Initializing varibales
	step_num = 0;

	cusparse_handle = 0;
	cusparse_descr = 0;

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

	if(step_num == 0)
		cout << "SIMULATION STARTED" << endl;

    if ( nCompt_ <= 0 )
        return;

    if ( !current_.size() )
    {
        current_.resize( channel_.size() );
    }
    step_num++;
#ifdef USE_CUDA
    //step_num++;
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
   	advanceChannels( info->dt );
   	calculateChannelCurrents();

   	/*
	if(step_num == 1) remove("current.csv");
	// Dealing with current
	ofstream currentFile("current.csv",ios::app);
	currentFile << inject_[3580].injectBasal << endl;
	*/
	updateMatrix();

	//HSolvePassive::forwardEliminate();
	//HSolvePassive::backwardSubstitute();

	pervasiveFlowSolverOpt();

	cudaMemcpy(d_Vmid, &(VMid_[0]), nCompt_*sizeof(double), cudaMemcpyHostToDevice);
	calculate_V_from_Vmid_wrapper(); // Avoing Vm memory transfer and using CUDA kernel

	advanceCalcium();
	advanceSynChans( info );
	sendValues( info );
	sendSpikes( info );
	//transfer_memory2cpu_cuda();

	// Checking for error after each time-step
	cudaCheckError();
#else
	advanceChannels( info->dt );
	calculateChannelCurrents();
	int solver_choice = 2; // 0-Moose , 1-Forward-flow, 2-Pervasive-flow

	/*
	string pPath = getenv("SOLVER_CHOICE");
	if(pPath.size() != 0){
		stringstream convert(pPath);
		convert >> solver_choice;
	}
	*/

	switch(solver_choice){
		case 0:
			updateMatrix();
			HSolvePassive::forwardEliminate();
			HSolvePassive::backwardSubstitute();
			break;
			break;
		case 1:
			// Using forward flow solution
			updateForwardFlowMatrix();
			forwardFlowSolver();
			break;
		case 2:
			// Using pervasive flow solution
			//updatePervasiveFlowMatrix();
			//pervasiveFlowSolver();
			updatePervasiveFlowMatrixOpt();
			pervasiveFlowSolverOpt();
			break;
	}

	advanceCalcium();
	advanceSynChans( info );
	sendValues( info );
	sendSpikes( info );

#endif

    externalCurrent_.assign( externalCurrent_.size(), 0.0 );
    
}

void HSolveActive::calculateChannelCurrents()
{
#ifdef USE_CUDA
	/*
	// TEMPORARY CODE
	GpuTimer timer;
	timer.Start();
		calculate_channel_currents_cuda_wrapper();
	timer.Stop();
	float time = timer.Elapsed();
	if(step_num < 20)
		cout << time << endl;
	*/

	calculate_channel_currents_cuda_wrapper();
	//cudaSafeCall(cudaMemcpy(&current_[0], d_current_, current_.size()*sizeof(CurrentStruct), cudaMemcpyDeviceToHost));
#else
	u64 startTime = getTime();
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
    u64 endTime = getTime();

    //if(step_num < 10)
    //		cout << ((endTime-startTime)/1000.0f)/time << " Calculate chan currents time " << time << " " << (endTime-startTime)/1000.0f << endl;
#endif
}

void HSolveActive::updateMatrix()
{

#ifdef USE_CUDA
	/*
	// Updates HS matrix and sends it to CPU
	if ( HJ_.size() != 0 )
		memcpy( &HJ_[ 0 ], &HJCopy_[ 0 ], sizeof( double ) * HJ_.size() );
	update_matrix_cuda_wrapper();
	*/

	// Copying initial matrix
	memcpy(qfull_mat.values, perv_mat_values_copy, qfull_mat.nnz*sizeof(double));
	update_perv_matrix_cuda_wrapper();

	// Updates CSR matrix and sends it to CPU.
	//update_csrmatrix_cuda_wrapper();

#else
	u64 startTime = getTime();
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
    u64 endTime = getTime();

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


void HSolveActive::updatePervasiveFlowMatrixOpt(){

	// Copying initial matrix
	memcpy(qfull_mat.values, perv_mat_values_copy, qfull_mat.nnz*sizeof(double));

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

		perv_dynamic[i] = per_mainDiag_passive[i] + GkSum;
		perv_dynamic[nCompt_+i] = V_[i] * compartment_[i].CmByDt + compartment_[i].EmByRm + GkEkSum;

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

	        perv_dynamic[nCompt_+ic] += (value.injectVarying + value.injectBasal);
	        value.injectVarying = 0.0;
	    }
	#endif

    vector< double >::iterator iec;
    for (unsigned int i = 0; i < nCompt_; i++)
    {
    	perv_dynamic[i] += externalCurrent_[2*i];
    	perv_dynamic[nCompt_+i] += externalCurrent_[2*i+1];
    }

    stage_ = 0;
}
void HSolveActive::pervasiveFlowSolverOpt(){
	//// Optimized pervasive flow solver
	//bool is_lower_triang;
	int elim_index = 0;
	for (int i = 0; i < nCompt_; ++i) {
		for (int j = qfull_mat.rowPtr[i]; j < upper_triang_offsets[i]; ++j) {
			// Eliminating an element
			int r1 = qfull_mat.colIndex[j];
			double scaling = qfull_mat.values[j]/perv_dynamic[2*r1];

			// Dealing with non-main diagonal
			for (int k = elim_rowPtr[elim_index]; k < elim_rowPtr[elim_index+1]; ++k){
				qfull_mat.values[eliminfo_r2[k]] -= (qfull_mat.values[eliminfo_r1[k]]* scaling);
			}

			// Dealing with main diagonal
			perv_dynamic[2*i] -= (qfull_mat.values[eliminfo_diag[elim_index]]*scaling);

			//Dealing with rhs
			perv_dynamic[2*i+1] -= (perv_dynamic[2*r1+1]*scaling);

			elim_index++;
		}
	}

	// Backward substitution
	bool is_upper_triang;
	double sum;
	for (int i = nCompt_-1; i >=0; --i) {
		sum = 0;
		for (int j = qfull_mat.rowPtr[i+1]-1; j >= upper_triang_offsets[i] ; --j) {
			sum += (qfull_mat.values[j]*VMid_[qfull_mat.colIndex[j]]);
		}

		VMid_[i] = (perv_dynamic[2*i+1]-sum)/perv_dynamic[2*i];
	}

	for (unsigned int i = 0; i < nCompt_; ++i) {
		V_[i] = 2*VMid_[i] - V_[i];
	}
	stage_ = 2;
}


void HSolveActive::advanceCalcium()
{
#ifdef USE_CUDA
	/* TEMPORARY CODE FOR Timings
	GpuTimer timer;
	timer.Start();
		advance_calcium_cuda_wrapper();
	timer.Stop();
	float time = timer.Elapsed();
	if(step_num < 10)	cout << "Advance calcium " << time << endl;
	*/

	advance_calcium_cuda_wrapper();

//#endif
#else

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

#endif
}

void HSolveActive::advanceChannels( double dt )
{

#ifdef USE_CUDA

    // Useful variables
    int num_comps = V_.size();

	if(!is_initialized){
		// TODO Move it to appropriate place

		// Initializing device Vm and Ca pools
		cudaSafeCall(cudaMemcpy(d_V, &(V_.front()), nCompt_ * sizeof(double), cudaMemcpyHostToDevice));
		cudaSafeCall(cudaMemcpy(d_state_, &(state_[0]), state_.size()*sizeof(double), cudaMemcpyHostToDevice));
		cudaSafeCall(cudaMemcpy(d_ca, &(ca_.front()), ca_.size()*sizeof(double), cudaMemcpyHostToDevice));

		// Setting up cusparse information. If SPMV approach is used then
		cusparseCreate(&cusparse_handle);

		// create and setup matrix descriptor
		cusparseCreateMatDescr(&cusparse_descr);
		cusparseSetMatType(cusparse_descr, CUSPARSE_MATRIX_TYPE_GENERAL);
		cusparseSetMatIndexBase(cusparse_descr, CUSPARSE_INDEX_BASE_ZERO);

		is_initialized = true;
	}

	// Calling the kernels
	//cudaSafeCall(cudaMemcpy(d_V, &(V_.front()), nCompt_ * sizeof(double), cudaMemcpyHostToDevice));
	//cudaSafeCall(cudaMemcpy(d_ca, &(ca_.front()), ca_.size()*sizeof(double), cudaMemcpyHostToDevice));

	get_lookup_rows_and_fractions_cuda_wrapper(dt); // Gets lookup values for Vm and Ca_.
	advance_channels_cuda_wrapper(dt); // Advancing fraction values.

	//cudaSafeCall(cudaMemcpy(&state_[0], d_state_, state_.size()*sizeof(double), cudaMemcpyDeviceToHost));

	if(step_num == 1){
		cout << channel_.size() << " " << channel_.size()*3 << " " << state_.size() << " extra % " << (state_.size()*100.0f)/(channel_.size()*3) << endl;
	}

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

#endif
      
}

#ifdef USE_CUDA

void HSolveActive::allocate_cpu_memory(){
	hits = new int[nCompt_]();
	stim_basal_values = new double[nCompt_](); // nCompt+1, because for non-stimulated compartments map id is zero.
	stim_comp_indices = new int[nCompt_](); // nCompt+1, because for non-stimulated compartments map id is zero.
	stim_map = new int[nCompt_]();
	num_stim_comp = 0;

	// Initializing elements in map to -1
	for (int i = 0; i < nCompt_; ++i) {
		stim_map[i] = -1;
	}

}

/* Used to allocate device memory on GPU for Hsolve variables */
void HSolveActive::allocate_hsolve_memory_cuda(){

	// Important numbers
	int num_compts = V_.size();
	int num_channels = channel_.size() ;
	int num_cmprsd_gates = state_.size();
	int num_Ca_pools = ca_.size();

	// LookUp Tables
	int V_table_size = vTable_.get_table().size();
	int Ca_table_size = caTable_.get_table().size();

	cudaMalloc((void **)&(d_V), num_compts * sizeof(double));
	cudaMalloc((void **)&(d_V_table), V_table_size * sizeof(double));

	cudaMalloc((void **)&(d_ca), num_Ca_pools * sizeof(double));
	cudaMalloc((void **)&(d_Ca_table), Ca_table_size * sizeof(double));

	//cudaMalloc((void**)&d_state_, num_cmprsd_gates*sizeof(double));

	// Channel related

	cudaMalloc((void**)&d_chan_instant, num_channels*sizeof(int));
	cudaMalloc((void**)&d_chan_modulation, num_channels*sizeof(double));
	cudaMalloc((void**)&d_chan_Gbar, num_channels*sizeof(double));
	cudaMalloc((void**)&d_chan_to_comp, num_channels*sizeof(double));
	cudaMalloc((void**)&d_chan_Gk, num_channels*sizeof(double));
	cudaMalloc((void**)&d_chan_Ek, num_channels*sizeof(double));
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

	// caConc_ Array of structures to structure of arrays related.
	cudaMalloc((void**)&d_CaConcStruct_c_, caConc_.size()*sizeof(double));
	cudaMalloc((void**)&d_CaConcStruct_CaBasal_, caConc_.size()*sizeof(double));
	cudaMalloc((void**)&d_CaConcStruct_factor1_, caConc_.size()*sizeof(double));
	cudaMalloc((void**)&d_CaConcStruct_factor2_, caConc_.size()*sizeof(double));
	cudaMalloc((void**)&d_CaConcStruct_ceiling_, caConc_.size()*sizeof(double));
	cudaMalloc((void**)&d_CaConcStruct_floor_, caConc_.size()*sizeof(double));

	// Hines Matrix related
	cudaMalloc((void**)&d_HS_, HS_.size()*sizeof(double));
	cudaMalloc((void**)&d_perv_dynamic, 2*nCompt_*sizeof(double));
	cudaMalloc((void**)&d_perv_static, nCompt_*sizeof(double));

	cudaMalloc((void**)&d_chan_colIndex, num_channels*sizeof(int));
	cudaMalloc((void**)&d_chan_rowPtr, (nCompt_+1)*sizeof(int));
	cudaMalloc((void**)&d_chan_x, nCompt_*sizeof(double));

	// Conjugate Gradient related
	cudaMalloc((void**)&d_Vmid, nCompt_*sizeof(double));

	// Intermediate data for computation
	cudaMalloc((void**)&d_V_rows, num_compts*sizeof(int));
	cudaMalloc((void**)&d_V_fractions, num_compts*sizeof(double));

	cudaMalloc((void**)&d_Ca_rows, num_Ca_pools*sizeof(int));
	cudaMalloc((void**)&d_Ca_fractions, num_Ca_pools*sizeof(double));

	// AdvanceCalcium related
	cudaMalloc((void**)&d_capool_values, h_catarget_channel_indices.size()*sizeof(double));
	cudaMalloc((void**)&d_capool_onex, h_catarget_channel_indices.size()*sizeof(double));

	// Optimized approach.
	cudaMalloc((void**)&d_state_, num_cmprsd_gates*sizeof(double));
	cudaMalloc((void**)&d_state_rowPtr, (num_channels+1)*sizeof(int));
	cudaMalloc((void**)&d_state_powers, num_cmprsd_gates*sizeof(double));
	cudaMalloc((void**)&d_state2chanId, num_cmprsd_gates*sizeof(int));
	cudaMalloc((void**)&d_state2column, num_cmprsd_gates*sizeof(int));

	//// Memory allocation for Variables of event based optimization for update_matrix method.
	cudaMalloc((void**)&d_stim_basal_values, nCompt_*sizeof(double));
	cudaMalloc((void**)&d_stim_comp_indices, nCompt_*sizeof(int));
	cudaMalloc((void**)&d_stim_map, nCompt_*sizeof(int));

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
	int num_cmprsd_gates = state_.size();
	int num_Ca_pools = ca_.size();

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

	// Optimized version
	double h_state_powers[num_cmprsd_gates];
	int h_state2chanId[num_cmprsd_gates];
	int h_state2column[num_cmprsd_gates];
	int* h_state_rowPtr = new int[num_channels+1]();

	// Gathering gate information and separating gates (with power < 0) , (with vm dependent) , (with ca dependent)
	int cmprsd_gate_index = 0; // If the logic is true cmprsd_gate_index value at the end of for loop = # of gates with powers > 0
	double h_gate_powers[3]; // Reusable array for holding powers of a channel
	for(unsigned int i=0;i<V_.size();i++){
		for(int j=chan_rowPtr[i]; j<chan_rowPtr[i+1]; j++){

			// Setting powers
			h_gate_powers[0] = channel_[j].Xpower_;
			h_gate_powers[1] = channel_[j].Ypower_;
			h_gate_powers[2] = channel_[j].Zpower_;

			for(int k=0;k<3;k++){
				if(h_gate_powers[k] > 0){

					// Collecting power of valid gate
					switch(k){
						case 0:
							h_state_powers[cmprsd_gate_index] = channel_[j].Xpower_;
							break;
						case 1:
							h_state_powers[cmprsd_gate_index] = channel_[j].Ypower_;
							break;
						case 2:
							h_state_powers[cmprsd_gate_index] = channel_[j].Zpower_;
							break;
					}

					// Collecting channel and column of valid gate
					h_state2chanId[cmprsd_gate_index] = j;
					h_state2column[cmprsd_gate_index] = column_[cmprsd_gate_index].column;

					// Partitioning of vm and ca dependent gates.
					if(k == 2 && caDependIndex_[j] != -1){
						h_cagate_indices.push_back(cmprsd_gate_index);
						h_cagate_capoolIds.push_back(ca_rowPtr[i] + caDependIndex_[j]);
					}else{
						h_vgate_compIds.push_back((int)chan2compt_[j]);
						h_vgate_indices.push_back(cmprsd_gate_index);
					}
					h_state_rowPtr[j] += 1;
					cmprsd_gate_index++; // cmprsd_gate_index is incremented only if power > 0 is found.
				}
			}
		}
	}
	assert(cmprsd_gate_index == num_cmprsd_gates);

	// Converting rowCounts to rowptr
	int csum = 0, ctemp;
	int zero_count = 0;
	for (int i = 0; i < num_channels+1; ++i) {
		ctemp = h_state_rowPtr[i];
		if(i < num_channels && h_state_rowPtr[i] == 0) zero_count++;
		h_state_rowPtr[i] = csum;
		csum += ctemp;
	}

	// Allocating memory (Optimized approach)
	cudaMalloc((void**)&d_vgate_indices, h_vgate_indices.size()* sizeof(int));
	cudaMalloc((void**)&d_vgate_compIds, h_vgate_compIds.size()* sizeof(int));

	cudaMalloc((void**)&d_cagate_indices, h_cagate_indices.size()* sizeof(int));
	cudaMalloc((void**)&d_cagate_capoolIds, h_cagate_capoolIds.size()* sizeof(int));

	// Transfering memory (Optimized approach)
	cudaMemcpy(d_state_rowPtr, h_state_rowPtr, (num_channels+1)*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_state_powers, h_state_powers, num_cmprsd_gates*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_state2chanId, h_state2chanId, num_cmprsd_gates*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_state2column, h_state2column, num_cmprsd_gates*sizeof(int), cudaMemcpyHostToDevice);

	cudaMemcpy(d_vgate_indices, &(h_vgate_indices[0]), h_vgate_indices.size()*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_vgate_compIds, &(h_vgate_compIds[0]), h_vgate_compIds.size()*sizeof(int), cudaMemcpyHostToDevice);

	cudaMemcpy(d_cagate_indices,&(h_cagate_indices[0]), h_cagate_indices.size()*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_cagate_capoolIds, &(h_cagate_capoolIds[0]), h_cagate_capoolIds.size()*sizeof(int), cudaMemcpyHostToDevice);


	// Allocating memory
	cudaMalloc((void**)&d_catarget_channel_indices, h_catarget_channel_indices.size()* sizeof(int));
	cudaMalloc((void**)&d_caActivation_values, ca_.size()* sizeof(double));

	cudaMalloc((void**)&d_capool_rowPtr, (num_Ca_pools+1)*sizeof(int));
	cudaMalloc((void**)&d_capool_colIndex, h_catarget_channel_indices.size()*sizeof(int));

	cudaMemcpy(d_catarget_channel_indices, &(h_catarget_channel_indices.front()), h_catarget_channel_indices.size()* sizeof(int), cudaMemcpyHostToDevice);


	// Current Data
	double h_Ek_temp[num_channels];
	for (int i = 0; i < num_channels; ++i) {
		h_Ek_temp[i] = current_[i].Ek;
	}
	cudaMemcpy(d_chan_Ek, h_Ek_temp, current_.size()*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_externalCurrent_, &(externalCurrent_.front()), 2 * num_compts * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_current_, &(current_.front()), current_.size()*sizeof(CurrentStruct), cudaMemcpyHostToDevice);
	cudaMemcpy(d_inject_, &(inject_.front()), inject_.size()*sizeof(InjectStruct), cudaMemcpyHostToDevice);
	cudaMemcpy(d_compartment_, &(compartment_.front()), compartment_.size()*sizeof(CompartmentStruct), cudaMemcpyHostToDevice);
	cudaMemcpy(d_caConc_, &(caConc_.front()), caConc_.size()*sizeof(CaConcStruct), cudaMemcpyHostToDevice);

	// Hines Matrix related
	cudaMemcpy(d_HS_, &(HS_.front()), HS_.size()*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_perv_static, per_mainDiag_passive, nCompt_*sizeof(double), cudaMemcpyHostToDevice);

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

	// caConc_ Array of structures to structure of arrays.
	double* temp_caconc = new double[caConc_.size()]();
	for (int i = 0; i < caConc_.size(); ++i) {
		temp_caconc[i] = caConc_[i].c_;
		cudaMemcpy(d_CaConcStruct_c_, temp_caconc, caConc_.size()*sizeof(double), cudaMemcpyHostToDevice);
	}

	for (int i = 0; i < caConc_.size(); ++i) {
		temp_caconc[i] = caConc_[i].CaBasal_;
		cudaMemcpy(d_CaConcStruct_CaBasal_, temp_caconc, caConc_.size()*sizeof(double), cudaMemcpyHostToDevice);
	}

	for (int i = 0; i < caConc_.size(); ++i) {
		temp_caconc[i] = caConc_[i].factor1_;
		cudaMemcpy(d_CaConcStruct_factor1_, temp_caconc, caConc_.size()*sizeof(double), cudaMemcpyHostToDevice);
	}

	for (int i = 0; i < caConc_.size(); ++i) {
		temp_caconc[i] = caConc_[i].factor2_;
		cudaMemcpy(d_CaConcStruct_factor2_, temp_caconc, caConc_.size()*sizeof(double), cudaMemcpyHostToDevice);
	}

	for (int i = 0; i < caConc_.size(); ++i) {
		temp_caconc[i] = caConc_[i].ceiling_;
		cudaMemcpy(d_CaConcStruct_ceiling_, temp_caconc, caConc_.size()*sizeof(double), cudaMemcpyHostToDevice);
	}

	for (int i = 0; i < caConc_.size(); ++i) {
		temp_caconc[i] = caConc_[i].floor_;
		cudaMemcpy(d_CaConcStruct_floor_, temp_caconc, caConc_.size()*sizeof(double), cudaMemcpyHostToDevice);
	}


	//// Memory transfer for Variables of event based optimization for update_matrix method.
	cudaMemcpy(d_stim_basal_values, stim_basal_values, nCompt_*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_stim_comp_indices, stim_comp_indices, nCompt_*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_stim_map, stim_map, nCompt_*sizeof(int), cudaMemcpyHostToDevice);

	/*
	// Writing load to file.
	ofstream load_file("umat_load.csv");
	load_file << "load" << endl;
	for (int i = 0; i < nCompt_; ++i) {
		load_file << chan_rowPtr[i+1]-chan_rowPtr[i] << endl;
	}
	*/
	/*
	ofstream chan_dist("state_dist.csv");
	for (int i = 0; i < nCompt_; ++i) {
		for (int j = chan_rowPtr[i]; j < chan_rowPtr[i+1]; ++j) {
			chan_dist << h_state_rowPtr[j+1]-h_state_rowPtr[j] << "," << i << endl;
		}
	}
	chan_dist.close();
	*/

	UPDATE_MATRIX_APPROACH = choose_update_matrix_approach();

	if(UPDATE_MATRIX_APPROACH == UPDATE_MATRIX_WPT_APPROACH){
		cout << "UM APPROACH : WPT" << endl;
	}else if(UPDATE_MATRIX_APPROACH == UPDATE_MATRIX_SPMV_APPROACH){
		cout << "UM APPROACH : SPMV" << endl;
	}else{
		// Future approaches, if any.
	}
}

void HSolveActive::transfer_memory2cpu_cuda(){
	cudaSafeCall(cudaMemcpy(&(ca_[0]), d_ca, ca_.size()*sizeof(double), cudaMemcpyDeviceToHost));
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
