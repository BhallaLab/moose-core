/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**   copyright (C) 2003-2007 Upinder S. Bhalla, Niraj Dudani and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _HSOLVE_ACTIVE_H
#define _HSOLVE_ACTIVE_H

#include "../basecode/header.h"
#include <set>
#include <limits> // Max and min 'double' values needed for lookup table init.
#include "../biophysics/CaConcBase.h"
#include "../biophysics/HHGate.h"
#include "../biophysics/ChanBase.h"
#include "../biophysics/ChanCommon.h"
#include "../biophysics/HHChannelBase.h"
#include "../biophysics/HHChannel.h"
#include "../biophysics/SpikeGen.h"
#include "HSolveUtils.h"
#include "HSolveStruct.h"
#include "HinesMatrix.h"
#include "HSolvePassive.h"

#include "CudaGlobal.h"
#include "RateLookup.h"
#include "Gpu_timer.h"

class HSolveActive: public HSolvePassive
{
    typedef vector< CurrentStruct >::iterator currentVecIter;

public:
    HSolveActive();

    void setup( Id seed, double dt );
    void step( ProcPtr info );			///< Equivalent to process
    void reinit( ProcPtr info );
#ifdef USE_CUDA
    LookupColumn * get_column_d();
#endif
protected:
    /**
     * Solver parameters: exposed as fields in MOOSE
     */

    /**
     * caAdvance_: This flag determines how current flowing into a calcium pool
     * is computed. A value of 0 means that the membrane potential at the
     * beginning of the time-step is used for the calculation. This is how
     * GENESIS does its computations. A value of 1 means the membrane potential
     * at the middle of the time-step is used. This is the correct way of
     * integration, and is the default way.
     */
    int                       caAdvance_;

    /**
     * vMin_, vMax_, vDiv_,
     * caMin_, caMax_, caDiv_:
     *
     * These are the parameters for the lookup tables for rate constants.
     * 'min' and 'max' are the boundaries within which the function is defined.
     * 'div' is the number of divisions between min and max.
     */
    double                    vMin_;
    double                    vMax_;
    int                       vDiv_;
    double                    caMin_;
    double                    caMax_;
    int                       caDiv_;

    /**
     * Internal data structures. Will also be accessed in derived class HSolve.
     */
    vector< CurrentStruct >   current_;			///< Channel current
    vector< double >          state_;			///< Fraction of gates open
    //~ vector< int >             instant_;
    vector< ChannelStruct >   channel_;			///< Vector of channels. Link
    ///< to compartment: chan2compt
    vector< SpikeGenStruct >  spikegen_;
    vector< SynChanStruct >   synchan_;
    vector< CaConcStruct >    caConc_;			///< Ca pool info
    vector< double >          ca_;				///< Ca conc in each pool
    vector< double >          caActivation_;	///< Ca current entering each
    ///< calcium pool
    vector< double* >         caTarget_;		///< For each channel, which
    ///< calcium pool is being fed?
    ///< Points into caActivation.
    LookupTable               vTable_;
    LookupTable               caTable_;
    vector< bool >            gCaDepend_;		///< Does the conductance
    ///< depend on Ca conc?
    vector< unsigned int >    caCount_;			///< Number of calcium pools in
    ///< each compartment
    vector< int >             caDependIndex_;	///< Which pool does each Ca
    ///< depdt channel depend upon?
    vector< LookupColumn >    column_;			///< Which column in the table
    ///< to lookup for this species
    vector< LookupRow >       caRowCompt_;      /**< Lookup row buffer.
		*   For each compartment, the lookup rows for calcium dependent
		*   channels are loaded into this vector before being used. The vector
		*   is then reused for the next compartment. This vector therefore has
		*   a size equal to the maximum number of calcium pools across all
		*   compartments. This is done in HSolveActive::advanceChannels */

    vector< LookupRow* >      caRow_;			/**< Points into caRowCompt.
		*   For each channel, points to the appropriate pool's LookupRow in the
		*   caRowCompt vector. This value is then used by the channel. Also
		*   happens in HSolveActive::advanceChannels */

    vector< int >             channelCount_;	///< Number of channels in each
    ///< compartment
    vector< currentVecIter >  currentBoundary_;	///< Used to designate compt
    ///< boundaries in the current_
    ///< vector.
    vector< unsigned int >    chan2compt_;		///< Index of the compt to
    ///< which a given (index)
    ///< channel belongs.
    vector< unsigned int >    chan2state_;		///< Converts a chnnel index to
    ///< a state index
    vector< double >          externalCurrent_; ///< External currents from
    ///< channels that HSolve
    ///< cannot internalize.
    vector< Id >              caConcId_;		///< Used for localIndex-ing.
    vector< Id >              channelId_;		///< Used for localIndex-ing.
    vector< Id >              gateId_;			///< Used for localIndex-ing.
    //~ vector< vector< Id > >    externalChannelId_;
    vector< unsigned int >    outVm_;			/**< VmOut info.
		*   Tells you which compartments have external voltage-dependent
		*   channels (if any), so that you can send out Vm values only in those
		*   places */
    vector< unsigned int >    outCa_;			/**< concOut info.
		*   Tells you which compartments have external calcium-dependent
		*   channels so that you can send out Calcium concentrations in only
		*   those compartments. */
#ifdef USE_CUDA    
		double total_time[10];
		int total_count;
    int                       current_ca_position;
    vector<ChannelData>		  channel_data_;
    ChannelData 			  * channel_data_d;
    void copy_to_device(double ** v_row_array, double * v_row_temp, int size);
    LookupColumn			  *column_d;
    int                       is_inited_;
	void copy_data(std::vector<LookupColumn>& column,
                             LookupColumn **            column_dd,
                             int *                      is_inited,
                             vector<ChannelData>&       channel_data,
                             ChannelData **             channel_data_dd,
                             const int                  x,
                             const int                  y,
                             const int                  z);

	// CUDA Passive Data
	vector<int> h_gate_expand_indices;
	vector<int> h_vgate_expand_indices;
	vector<int> h_vgate_compt_indices;
	vector<int> h_cagate_expand_indices;
	vector<int> h_cagate_capool_indices;

	vector<int> h_catarget_channel_indices; // Stores the indices of channel which are ca targets in order
	vector<int> h_catarget_capool_indices; // Store the index of calcium pool

	// LookUp Tables
	double* d_V_table;
	double* d_Ca_table;

	// Gate related
	double* d_gate_values; // Values of x,y,x for all channels.
	double* d_gate_powers; // Powers of x,y,z for all channels.
	int* d_gate_columns; // Corresponding columns of lookup tables
	int* d_gate_ca_index; // -1 -> V_lookup , (>0) -> Ca_lookup

	int* d_gate_expand_indices; // Srotes the indices of gates for which power is > 0
	int* d_vgate_expand_indices; // Stores the indices of gates using vmtable in gates array.
	int* d_vgate_compt_indices; // Stores the compartment index for this gate.
	int* d_cagate_expand_indices; // Stores the indices of gates using cmtable in gates array.
	int* d_cagate_capool_indices; // Stores the indices of calcium pools in ca_ array.

	int* d_catarget_channel_indices;
	int* d_catarget_capool_indices;
	double* d_caActivation_values; // Stores ca currents for that pool.

	int* d_capool_rowPtr;
	int* d_capool_colIndex;
	double* d_capool_values;
	double* d_capool_onex;


	double* d_state_;
  //int* d_gate_to_chan; // Not needed as we store 3 gates(x,y,z) for each channel.

	// Channel related
	int* d_chan_instant;
	double* d_chan_modulation;
	double* d_chan_Gbar;
	int* d_chan_to_comp; // Which compartment does a Channel belong to.

	double* d_chan_Gk;
	double* d_chan_GkEk;
	double* d_comp_Gksum;
	double* d_comp_GkEksum;
	double* d_externalCurrent_;
	CurrentStruct* d_current_;
	InjectStruct* d_inject_;
	CompartmentStruct* d_compartment_;
	CaConcStruct* d_caConc_;

	// Hines Matrix related
	double* d_HS_;

	double* d_chan_x;
	int* d_chan_colIndex;
	int* d_chan_rowPtr;

	// Conjugate Gradient based GPU solver
	double* d_Vmid, *d_p, *d_Ax, *d_r, *d_x;

	// LU based CPU solver
	csrluInfoHost_t infoA;
	size_t internalDataInBytes;
	size_t workspaceInBytes;

	double* internalBuffer;
	double* workspaceBuffer;

	/* Get handle to the CUBLAS context */
	cublasHandle_t cublas_handle = 0;
	cublasStatus_t cublasStatus;

	/* Get handle to the CUSPARSE context */
	cusparseHandle_t cusparse_handle = 0;
	cusparseMatDescr_t cusparse_descr = 0;

	/* Get handle for CUSOLVER context*/
	cusolverSpHandle_t cusolver_handle = 0;

	// Compartment related

	// CUDA Active Permanent data
	double* d_V;
	double* d_ca;


	// CUDA Active helper data
	int* d_V_rows;
	double* d_V_fractions;
	int* d_Ca_rows;
	double* d_Ca_fractions;
	int* d_temp_keys;
	double* d_temp_values;

	int num_comps_with_chans = 0; // Stores number of compartments with >=1 channels.

	// temp code
	bool is_initialized = false;
#endif

	int num_time_prints = 0;
	int num_um_prints = 0;
	int num_profile_prints = 0;
	int step_num = 0;

    static const int INSTANT_X;
    static const int INSTANT_Y;
    static const int INSTANT_Z;
private:
    /**
     * Setting up of data structures: Defined in HSolveActiveSetup.cpp
     */
    void readHHChannels();
    void readGates();
    void readCalcium();
    void readSynapses();
    void readExternalChannels();
    void createLookupTables();
    void manageOutgoingMessages();

    void cleanup();

    /**
     * Reinit code: Defined in HSolveActiveSetup.cpp
     */
    void reinitSpikeGens( ProcPtr info );
    void reinitCompartments();
    void reinitCalcium();
    void reinitChannels();

    /**
     * Integration: Defined in HSolveActive.cpp
     */
    void calculateChannelCurrents();
    void updateMatrix();
    void forwardEliminate();
    void backwardSubstitute();
    void advanceCalcium();
    void advanceChannels( double dt );
    void advanceSynChans( ProcPtr info );
    void sendSpikes( ProcPtr info );
    void sendValues( ProcPtr info );

#ifdef USE_CUDA
    // Hsolve GPU set up kernels
    void allocate_hsolve_memory_cuda();
    void copy_table_data_cuda();
    void copy_hsolve_information_cuda();
    void transfer_memory2cpu_cuda();

    void get_lookup_rows_and_fractions_cuda_wrapper(double dt);
    void advance_channels_cuda_wrapper(double dt);
    void get_compressed_gate_values_wrapper();

    void calculate_channel_currents_cuda_wrapper();

    void update_matrix_cuda_wrapper();
    void update_csrmatrix_cuda_wrapper();

    void hinesMatrixSolverWrapper();

    void advance_calcium_cuda_wrapper();

	void advanceChannel_gpu(
    double *                          v_row,
    vector<double>&                   caRow,
    LookupColumn                    * column,                                           
    LookupTable&                     vTable,
    LookupTable&                     caTable,                       
    double                          * istate,
    ChannelData                     * channel,
    double                          dt,
    int                             set_size,
    int                             channel_size,
    int                             num_of_compartment,
    float							&kernel_time
    );
#endif

};

#endif // _HSOLVE_ACTIVE_H
