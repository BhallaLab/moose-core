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

class HSolveActive: public HSolvePassive
{
    typedef vector< CurrentStruct >::iterator currentVecIter;

public:
    HSolveActive();

    void setup( Id seed, double dt );
    void step( ProcPtr info );			///< Equivalent to process
    void reinit( ProcPtr info );
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
    int step_num = 0;
#ifdef USE_CUDA    
    //int step_num;

    // Optimized data
	vector<int> h_vgate_indices;
	vector<int> h_vgate_compIds;

	vector<int> h_cagate_indices;
	vector<int> h_cagate_capoolIds;

	vector<int> h_catarget_channel_indices; // Stores the indices of channel which are ca targets in order
	vector<int> h_catarget_capool_indices; // Store the index of calcium pool

	// LookUp Tables
	double* d_V_table;
	double* d_Ca_table;

	// Optimized version
	double* d_state_; // state of gate values, such as (m,h,n...)
	double* d_state_powers; // Powers of gate values, such as (3 for m^3)
	int* d_state_rowPtr; // RowPtr on valid gates where rows = num_channels
	int* d_state2chanId; // Channel Id to which this gate belongs to.
	int* d_state2column; // Corresponding column in lookup table.
	int* d_vgate_indices; // Set of gates that are voltage dependent
	int* d_vgate_compIds; // Corresponding compartment id of voltage dependent gate

	int* d_cagate_indices; // Set of gates that are calcium dependent.
	int* d_cagate_capoolIds; // Corresponding calcium pool id of calcium dependent gate

	// advanceCalcium related.
	int* d_catarget_channel_indices;
	double* d_caActivation_values; // Stores ca currents for that pool.

	int* d_capool_rowPtr;
	int* d_capool_colIndex;
	double* d_capool_values;
	double* d_capool_onex;


	// Channel related
	int* d_chan_instant;
	double* d_chan_modulation;
	double* d_chan_Gbar;
	int* d_chan_to_comp; // Which compartment does a Channel belong to.

	double* d_chan_Gk;
	double* d_chan_Ek;
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
	double* d_perv_dynamic;
	double* d_perv_static;

	double* d_chan_x;
	int* d_chan_colIndex;
	int* d_chan_rowPtr;
	int UPDATE_MATRIX_APPROACH;
	int UPDATE_MATRIX_WPT_APPROACH = 0;
	int UPDATE_MATRIX_SPMV_APPROACH = 1;

	// Conjugate Gradient based GPU solver
	double* d_Vmid;

	/* Get handle to the CUSPARSE context */
	cusparseHandle_t cusparse_handle;
	cusparseMatDescr_t cusparse_descr;

	// caConc_ Array of structures to structure of arrays.
	double* d_CaConcStruct_c_; // Dynamic array
	double* d_CaConcStruct_CaBasal_, *d_CaConcStruct_factor1_, *d_CaConcStruct_factor2_, *d_CaConcStruct_ceiling_, *d_CaConcStruct_floor_; // Static array

	// CUDA Active Permanent data
	double* d_V;
	double* d_ca;


	// CUDA Active helper data
	int* d_V_rows;
	double* d_V_fractions;
	int* d_Ca_rows;
	double* d_Ca_fractions;

	bool is_initialized; // Initializing device memory data
#endif

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

    void updateForwardFlowMatrix();
    void forwardFlowSolver();

    void updatePervasiveFlowMatrix();
    void pervasiveFlowSolver();

    void updatePervasiveFlowMatrixOpt();
    void pervasiveFlowSolverOpt();

#ifdef USE_CUDA
    // Hsolve GPU set up kernels
    void allocate_hsolve_memory_cuda();
    void copy_table_data_cuda();
    void copy_hsolve_information_cuda();
    void transfer_memory2cpu_cuda();

    void get_lookup_rows_and_fractions_cuda_wrapper(double dt);
    void advance_channels_cuda_wrapper(double dt);

    void calculate_channel_currents_cuda_wrapper();

    void update_matrix_cuda_wrapper();
    void update_perv_matrix_cuda_wrapper();
    void update_csrmatrix_cuda_wrapper();
    int choose_update_matrix_approach();

    void calculate_V_from_Vmid_wrapper();

    void advance_calcium_cuda_wrapper();

#endif

};

#endif // _HSOLVE_ACTIVE_H
