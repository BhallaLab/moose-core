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
using namespace thrust;

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
#endif


    // Default lookup table size
    //~ vDiv_ = 3000;    // for voltage
    //~ caDiv_ = 3000;   // for calcium
}

//////////////////////////////////////////////////////////////////////
// Solving differential equations
//////////////////////////////////////////////////////////////////////

/*
 * A debug function to profile the performance of selected modules.
 */
void update_info(double* time, int func, int elapsed, int count, double dt)
{
	time[func] += elapsed / 1000.0f;
	char * str;
	if(count >= (0.2/dt) - 1)
	{
		switch(func)
		{
			case 0: str = "advanceChannels";break;
			case 1: str = "calculateChannelCurrents";break;
			case 2: str = "updateMatrix";break;
			case 3: str = "forwardEliminate";break;
			case 4: str = "backwardSubstitute";break;
			case 5: str = "advanceCalcium";break;
			case 6: str = "advanceSynChans";break;
			case 7: str = "sendValues";break;
			case 8: str = "sendSpikes";break;
			case 9: str = "externalCurrent_.assign";break;
			default: str = "Unkown";break;
		}
		printf("Function %s takes %fs.\n", str, time[func]/ 1000.0f);
	}
}
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
#endif
    

#ifdef PROFILE_CUDA
    u64 start_time, end_time;
    start_time = getTime();
#endif
    
    advanceChannels( info->dt );
    
#ifdef PROFILE_CUDA
    end_time = getTime();
    update_info(total_time, 0, end_time - start_time, total_count ,info->dt);
    start_time = end_time;
#endif
    
    calculateChannelCurrents();

#ifdef PROFILE_CUDA
    end_time = getTime();
    update_info(total_time, 1, end_time - start_time, total_count ,info->dt);
    start_time = end_time;   
#endif
    
    updateMatrix();

#ifdef PROFILE_CUDA
    end_time = getTime();
    update_info(total_time, 2, end_time - start_time, total_count ,info->dt);
    start_time = end_time;   
#endif
        
    HSolvePassive::forwardEliminate();
    
#ifdef PROFILE_CUDA
    end_time = getTime();
    update_info(total_time, 3, end_time - start_time, total_count ,info->dt);
    start_time = end_time;   
#endif
        
    HSolvePassive::backwardSubstitute();
    
#ifdef PROFILE_CUDA
    end_time = getTime();
    update_info(total_time, 4, end_time - start_time, total_count ,info->dt);
    start_time = end_time;   
#endif
        
    advanceCalcium();
    
#ifdef PROFILE_CUDA
    end_time = getTime();
    update_info(total_time, 5, end_time - start_time, total_count ,info->dt);
    start_time = end_time;   
#endif
        
    advanceSynChans( info );
    
#ifdef PROFILE_CUDA
    end_time = getTime();
    update_info(total_time, 6, end_time - start_time, total_count ,info->dt);
    start_time = end_time;   
#endif
    
    sendValues( info );
    
#ifdef PROFILE_CUDA
    end_time = getTime();
    update_info(total_time, 7, end_time - start_time, total_count ,info->dt);
    start_time = end_time;   
#endif
        
    sendSpikes( info );
     
#ifdef PROFILE_CUDA
    end_time = getTime();
    update_info(total_time, 8, end_time - start_time, total_count ,info->dt);
    start_time = end_time;   
#endif
       

    externalCurrent_.assign( externalCurrent_.size(), 0.0 );
    
#ifdef PROFILE_CUDA
    end_time = getTime();
    update_info(total_time, 9, end_time - start_time, total_count ,info->dt);
#endif

}

void HSolveActive::calculateChannelCurrents()
{
#ifdef USE_CUDA
	//printf("Calculating Channel Currents\n");
	GpuTimer currentTimer;

	currentTimer.Start();
		calculate_channel_currents_cuda_wrapper();
		cudaSafeCall(cudaMemcpy(&current_[0], d_current_, current_.size()*sizeof(CurrentStruct), cudaMemcpyDeviceToHost));
	currentTimer.Stop();

	/*
	cudaCheckError(); // Making sure no CUDA error occured
	for(unsigned int i=0;i<current_.size();i++)
		current_[i].Gk = h_chan_Gk[i];
	*/

	if(num_time_prints > 0){
		printf("curr,%lf\n", currentTimer.Elapsed());
		num_time_prints--;
	}

#else
	u64 cpu_current_start = getTime();

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

    u64 cpu_current_end = getTime();

    if(num_time_prints > 0)
    	printf("curr,%lf\n",(cpu_current_end-cpu_current_start)/1000.0f);

    /*
    double error = 0;
    for(unsigned int i=0;i<current_.size();i++){
    	printf("%lf %lf \n", current_[i].Gk, h_chan_Gk[i]);
    	error += (current_[i].Gk - h_chan_Gk[i]);
    }
    getchar();
    printf("%lf error in channel currents\n", error);
    */

#endif
}

void HSolveActive::updateMatrix()
{

	/*
	 * Copy contents of HJCopy_ into HJ_. Cannot do a vector assign() because
	 * iterators to HJ_ get invalidated in MS VC++
	 * TODO Is it needed to do a memcpy each time?
	 */
	if ( HJ_.size() != 0 )
		memcpy( &HJ_[ 0 ], &HJCopy_[ 0 ], sizeof( double ) * HJ_.size() );

#ifdef USE_CUDA
	/*
	if(num_um_prints > 0){
		// Printing external sums
		double extsum1 = 0;
		double extsum2 = 0;
		int nzero_count1 = 0;
		int nzero_count2 = 0;
		for(int i=0;i<externalCurrent_.size();i=i+2){
			extsum1 += externalCurrent_[i];
			extsum2 += externalCurrent_[i+1];

			if(externalCurrent_[i] != 0)
				nzero_count1++;
			if(externalCurrent_[i+1] != 0)
				nzero_count2++;


		}
		if(nzero_count1 > 0 || nzero_count2 > 0)
			printf("sum1 %lf sum2 %lf count1 %d count2 %d \n",extsum1,extsum2, nzero_count1, nzero_count2);

		// Printing inject_
		double injsum1 = 0;
		double injsum2 = 0;
		nzero_count1 = 0;
		nzero_count2 = 0;
		for(int i=0;i<inject_.size();i++){
			injsum1 += inject_[i].injectVarying;
			injsum2 += inject_[i].injectBasal;

			if(inject_[i].injectVarying != 0) nzero_count1++;
			if(inject_[i].injectBasal != 0) {
				//printf("%d comaprtment inject basal %lf\n", i, inject_[i].injectBasal*100000);
				nzero_count2++;
			}

		}

		if(nzero_count1 > 0)
			printf("varying %lf base %lf total %lf sum1 %d sum2 %d\n",injsum1, injsum2, injsum1+injsum2, nzero_count1, nzero_count2);

		num_um_prints--;
	}
	*/

	GpuTimer gpuTimer;

	gpuTimer.Start();
		update_matrix_cuda_wrapper();
		cudaCheckError();
	gpuTimer.Stop();

	if(num_um_prints > 0){
		printf("%lf\n",gpuTimer.Elapsed());
		num_um_prints--;
	}

	// Use thrust reduce_by_key to find GkSum and GKEkSum
	// Use Gksum, GkEksum,
#else
	// CPU PART
	u64 cpu_start = getTime();

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

    u64 cpu_end = getTime();

    if(num_um_prints > 0){
		printf("um,%lf\n", (cpu_end-cpu_start)/1000.0f);
		num_um_prints--;
	}
#endif
    stage_ = 0;    // Update done.
}

void HSolveActive::advanceCalcium()
{
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
}

void HSolveActive::advanceChannels( double dt )
{

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

#ifdef USE_CUDA

#ifdef TEST_CORRECTNESS
    double* gate_values_after_cuda = new double[h_gate_expand_indices.size()]();
#endif

    // Useful numbers
    int num_comps = V_.size();
    int num_gates = channel_.size()*3;

    GpuTimer tejaTimer;

    tejaTimer.Start();
    // ----------------------------------------------------------------------------
    cudaMemcpy(d_V, &(V_.front()), nCompt_ * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ca, &(ca_.front()), ca_.size()*sizeof(double), cudaMemcpyHostToDevice);

	if(!init_gate_values){
		// TODO Move it to appropriate place
		printf("INITIALIZING Gate Values\n");
		double* test_gate_values = new double[num_gates]();
		// Copying from state_ to test_gate_values
		for(int i=0;i<h_gate_expand_indices.size();i++){
			test_gate_values[h_gate_expand_indices[i]] = state_[i];
		}
		cudaMemcpy(d_gate_values, test_gate_values, num_gates*sizeof(double), cudaMemcpyHostToDevice);

		init_gate_values = true;

		// Setting up cusparse information
		cusparseCreate(&cusparse_handle);

		// create and setup matrix descriptors A, B & C
		cusparseCreateMatDescr(&cusparse_descr);
		cusparseSetMatType(cusparse_descr, CUSPARSE_MATRIX_TYPE_GENERAL);
		cusparseSetMatIndexBase(cusparse_descr, CUSPARSE_INDEX_BASE_ZERO);

		/*
		set<int> comps;
		for(int i=0;i<channel_.size();i++){
			comps.insert(chan2compt_[i]);
		}

		for (set<int>::iterator i = comps.begin(); i != comps.end(); i++) {
		   cout << *i << endl;
		}

		cout << nCompt_ << " " << comps.size() << endl;

		int count = 0;
		for (int i = 0; i < channelCount_.size(); ++i) {
			if(channelCount_[i] == 0)
				count++;
		}

		cout << "# of compartments with zero channels " << count << endl;
		*/

	}

	// ----------------------------------------------------------------------------

	get_lookup_rows_and_fractions_cuda_wrapper(dt);
	advance_channels_cuda_wrapper(dt); // Calling kernel
	get_compressed_gate_values_wrapper(); // Getting values of new state
#ifdef TEST_CORRECTNESS
	cudaSafeCall(cudaMemcpy(gate_values_after_cuda, d_state_, state_.size()*3*sizeof(double), cudaMemcpyDeviceToHost));
#else
	cudaSafeCall(cudaMemcpy(&state_[0], d_state_, state_.size()*sizeof(double), cudaMemcpyDeviceToHost));
#endif
	cudaCheckError(); // Making sure no CUDA errors occured
	tejaTimer.Stop();

	if(num_time_prints > 0){
		printf("chan,%lf\n", tejaTimer.Elapsed());
		num_time_prints--;
	}

	/*
	// Checking correctness of get_lookup_rows_and_fractions_cuda Kernel
	double test_V[num_comps];
	int test_rows[num_comps];
	double test_fracs[num_comps];

	cudaMemcpy(test_V, d_V, num_comps*sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(test_rows, d_V_rows, num_comps*sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(test_fracs, d_V_fractions, num_comps*sizeof(double), cudaMemcpyDeviceToHost);

	*/

#ifdef TEST_CORRECTNESS
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


	// Printing values of cuda call and cpu call
	float error = 0;
	for(int i=0;i<state_.size();i++){
		error += gate_values_after_cuda[i]-state_[i];
		printf("%lf %lf\n",state_[i], gate_values_after_cuda[i]);
	}
	printf("Error %f\n",error);
	getchar();


#endif

#else

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

    if(num_time_prints > 0){
    	printf("chan,%lf\n", (cpu_advchan_end-cpu_advchan_start)/1000.0f);
    	num_time_prints--;
    }

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

	// CPU related
	h_chan_Gk = (double*) malloc(num_channels * sizeof(double));


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


	// Compartment related

	// Hines Matrix related
	cudaMalloc((void**)&d_HS_, HS_.size()*sizeof(double));
	cudaMalloc((void**)&d_HS_1, nCompt_*sizeof(double));
	cudaMalloc((void**)&d_HS_2, nCompt_*sizeof(double));

	cudaMalloc((void**)&d_chan_colIndex, num_channels*sizeof(int));
	cudaMalloc((void**)&d_chan_rowPtr, (nCompt_+1)*sizeof(int));
	cudaMalloc((void**)&d_chan_x, nCompt_*sizeof(double));



	// Intermediate data for computation

	cudaMalloc((void**)&d_V_rows, num_compts*sizeof(int));
	cudaMalloc((void**)&d_V_fractions, num_compts*sizeof(double));

	cudaMalloc((void**)&d_Ca_rows, num_Ca_pools*sizeof(int));
	cudaMalloc((void**)&d_Ca_fractions, num_Ca_pools*sizeof(double));

	cudaMalloc((void**)&d_temp_keys, num_compts*sizeof(int));
	cudaMalloc((void**)&d_temp_values, num_compts*sizeof(double));

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


	cudaMemcpy(d_gate_values, h_gate_values, num_gates * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_gate_powers, h_gate_powers, num_gates * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_gate_columns, h_gate_columns, num_gates * sizeof(int), cudaMemcpyHostToDevice);

	cudaMemcpy(d_gate_ca_index, h_gate_ca_index, num_gates * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_gate_expand_indices, &(h_gate_expand_indices.front()), h_gate_expand_indices.size()* sizeof(int), cudaMemcpyHostToDevice);

	cudaMemcpy(d_vgate_expand_indices, &(h_vgate_expand_indices.front()), h_vgate_expand_indices.size()* sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_vgate_compt_indices, &(h_vgate_compt_indices.front()), h_vgate_compt_indices.size()* sizeof(int), cudaMemcpyHostToDevice);

	cudaMemcpy(d_cagate_expand_indices, &(h_cagate_expand_indices.front()), h_cagate_expand_indices.size()* sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_cagate_capool_indices, &(h_cagate_capool_indices.front()), h_cagate_capool_indices.size()* sizeof(int), cudaMemcpyHostToDevice);

	// Current Data
	cudaMemcpy(d_externalCurrent_, &(externalCurrent_.front()), 2 * num_compts * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_current_, &(current_.front()), current_.size()*sizeof(CurrentStruct), cudaMemcpyHostToDevice);
	cudaMemcpy(d_inject_, &(inject_.front()), inject_.size()*sizeof(InjectStruct), cudaMemcpyHostToDevice);
	cudaMemcpy(d_compartment_, &(compartment_.front()), compartment_.size()*sizeof(CompartmentStruct), cudaMemcpyHostToDevice);

	// Hines Matrix related
	cudaMemcpy(d_HS_, &(HS_.front()), HS_.size()*sizeof(double), cudaMemcpyHostToDevice);

	double* temp_hs_1 = new double[nCompt_]();
	double* temp_hs_2 = new double[nCompt_]();

	for(int i=0;i<nCompt_;i++){
		temp_hs_1[i] = HS_[4*i+1];
		temp_hs_2[i] = HS_[4*i+2];
	}

	cudaMemcpy(d_HS_1, temp_hs_1 , nCompt_*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_HS_2, temp_hs_2 , nCompt_*sizeof(double), cudaMemcpyHostToDevice);

	int* chan_colIndex = new int[num_channels]();
	int* chan_x = new int[nCompt_];

	std::fill_n(chan_x, nCompt_, 1);

	// Filling column indices
	for(int i=0;i<nCompt_;i++){
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
