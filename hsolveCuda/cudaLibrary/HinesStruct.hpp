#ifndef HINESSTRUCT_HPP
#define HINESSTRUCT_HPP

//struct __align__(16) HinesStruct{
struct HinesStruct{


	int type;

	ucomp *ucompMemory;
	ucomp *mulListDest;
	ucomp *mulListComp;
	int mulListSize;

	ftype *memoryS;
	ftype *memoryE;

	/**
	 * Hines matrix and auxiliary data
	 */
	ftype *triangList;

	ftype *leftList;
	ucomp *leftListLine;
	ucomp *leftListColumn;
	ucomp *leftStartPos;
	int leftListSize;

	ftype *rhsM;
	ftype *vmList;
	ftype *vmTmp;
	ftype *mulList;


	/**
	 * Information about the compartments
	 */
	int nComp;
	int nNeurons;
	ftype *Cm;
	ftype *Rm;
	ftype *Ra;
	ftype vRest;
	ftype dx;

	ftype *curr;   // Injected current

	int triangAll;

	/******************************************************************
	 * Active Channels (NEW)
	 ******************************************************************/

	ftype *active;    // Active current (size = # of compartments)
	ftype *gateState; // opening of the gates, indexed by gatePos in the channelInfo (size = # of gates)

	ucomp *channelInfo; // nGates(0) comp(1) gatePos(3)
	ftype *channelEk;
	ftype *channelGbar;

	ucomp *gateInfo;    // gatePower(0): function alpha (1) and function beta (2)
	ftype *gatePar;   // parameters of alpha (A, B, V0) (0,1,2) and beta (3,4,5) functions

	int nChannels;
	int nGatesTotal;

	int compListSize; // Number of compartments with active channels
	ucomp *compList;  // List of compartments with active channels
	ftype *eLeak;     // contains the eLEak of the active compartments

	ftype *gActive; // Contains the active channel conductances.

	/******************************************************************
	 * Synaptic Channels (NEW)
	 ******************************************************************/

	int synapseListSize;	// *

	ftype *synConstants;	// shared *

	ftype *synState; 		// exclusive *

	ucomp *synapseCompList; // shared *
	ucomp *synapseTypeList; // shared *

	int activationListSize; // *
	ftype *activationList; 	// exclusive *
	ucomp *activationListPos; // exclusive *


	/******************************************************************
	 * Generated spikes
	 ******************************************************************/

	// Contains the time of the spikes generated in the current execution block
	ftype *spikeTimes;
	int spikeTimeListSize;
	// Number of spikes generated in the current block (not a vector, just a pointer to a memory location)
	ucomp *nGeneratedSpikes;

	// Contains the time of the last spike generated on the neuron
	ftype lastSpike;
	ftype threshold; // in mV
	ftype minSpikeInterval; // in mV

	/**********************************************************************/


	/**
	 * Holds the results that will be copied to the CPU
	 */
	ftype *vmTimeSerie;

	/**
	 * Simulation information
	 */
	int currStep;
	ftype dt;

};

#endif
