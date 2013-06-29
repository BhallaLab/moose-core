/*
 * ActiveChannels.h
 *
 *  Created on: 16/06/2009
 *      Author: rcamargo
 */

#ifndef ACTIVECHANNELS_H_
#define ACTIVECHANNELS_H_

#define EXPONENTIAL 0   // A exp((v-V0)/B)
#define SIGMOID 1 		// A / (exp((v-V0)/B) + 1)
#define LINOID 2        // A (v-V0) / (exp((v-V0)/B) - 1)

#define N_GATE_FIELDS 3

#define GATE_POWER 0
#define ALPHA_FUNCTION 1
#define BETA_FUNCTION 2

#define N_CHANNEL_FIELDS 3

#define CH_NGATES 0
#define CH_COMP 1
#define CH_GATEPOS 2

#define N_GATE_FUNC_PAR 6

#define A_A  0
#define A_B  1
#define A_V0 2

#define B_A  3
#define B_B  4
#define B_V0 5

#include "Definitions.hpp"

class ActiveChannels {

	public:

	ftype dt;
	int nActiveComp;
	ucomp *activeCompList;

	ftype *vmList;

	ActiveChannels(ftype dt, ftype *vmListNeuron_, int nComp);
	virtual ~ActiveChannels();


	void setActiveChannels(int nActiveComp_, ucomp *activeCompList_);

	int getCompListSize () { return nActiveComp; }
	ucomp *getCompList () { return activeCompList; }


	/**
	 * New implementation
	 */
	ucomp *ucompMem;
	int ucompMemSize;
	ftype *ftypeMem;
	int ftypeMemSize;

	ucomp *channelInfo; // nGates(0) comp(1) gatePos(3)
	ftype *channelEk;
	ftype *channelGbar;

	int nChannels;    //
	int nGatesTotal;  // not used by the class, only by the CUDA implementation

	ucomp *gateInfo;    // gatePower(0): function alpha (1) and function beta (2)
	ftype *gatePar;   // parameters of alpha (A, B, V0) (0,1,2) and beta (3,4,5) functions

	ftype *gateState; // opening of the gates, indexed by gatePos in the channelInfo

	int nComp;
	ftype *gActive; // Contains the sum of the active conductances.

	ftype *eLeak; // contains the eLEak of the active compartments


	ftype getActiveConductances(int activeComp) {return gActive[activeComp];}

	void evaluateCurrents( ftype *Rm, ftype *active );
	void evaluateGates();

	void createChannelList (int nChannels, ucomp *nGates, ucomp *comp, ftype *channelEk, ftype *gBar, ftype *eLeak, int nActiveComp_, ucomp *activeCompList_);

	void setGate (int channel, int gate, ftype state, ucomp gatePower,
			ucomp alpha, ftype alphaA, ftype alphaB, ftype alphaV0,
			ucomp beta, ftype betaA, ftype betaB, ftype betaV0);
};

#endif /* ACTIVECHANNELS_H_ */
