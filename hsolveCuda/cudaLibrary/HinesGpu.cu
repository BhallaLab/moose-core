/*
 * HinesGpu.cu
 *
 *  Created on: 06/06/2009
 *      Author: rcamargo
 */

/**
 * TODO: otimizar
 * - acessar sTriangList com POS
 * - Ver acessos à memória global
 */

/**
 * TODO:
 * - Otimizar (usar memória compartilhada)
 * - Otimizar (reduzir número de alocações de memória para copiar lista de spikes)
 */

//extern "C" {
#include "HinesMatrix.hpp"
#include "PlatformFunctions.hpp"
#include "HinesStruct.hpp"
#include <cassert>

#include <cuda.h> // Necessary to allow better eclipse integration
#include <cuda_runtime_api.h> // Necessary to allow better eclipse integration
#include <device_launch_parameters.h> // Necessary to allow better eclipse integration
#include <device_functions.h> // Necessary to allow better eclipse integration


//#define POS(i) (i) + nComp*threadIdx.x
#define POS(i) (i)*blockDim.x+threadIdx.x
//#define POS1(i) (i) + leftListSize*threadIdx.x
#define POS1(i) (i)*blockDim.x + threadIdx.x

#define NLOCALSPIKES 16 // 16

/***************************************************************************
 * This part is executed in every integration step
 ***************************************************************************/

__device__ void evaluateSynapticCurrentsNew( HinesStruct *hList, ftype *active, ftype *vmList,
		ftype currTime,	int synapseListSize, ftype *synConstants, ftype *synState, ucomp *synapseCompList,
		int activationListSize, ftype *activationList, ucomp *activationListPos ) {

	int neuron = blockIdx.x * blockDim.x + threadIdx.x;

	for (int syn=0; syn < synapseListSize; syn++) {
		int currPos = (syn * activationListSize) + activationListPos[syn];
		currPos = currPos * hList[neuron].nNeurons + neuron; // global interleave lList
		ftype activation = activationList[ currPos ];
		activationList[ currPos ] = 0;

		activationListPos[syn] = (activationListPos[syn] + 1) % activationListSize;

		int synComp = synapseCompList[syn];

		// TODO: problem is in one of the lines below
		synState[SYN_STATE_X] = synConstants[SYN_MOD] * activation * synConstants[SYN_X1] + synState[SYN_STATE_X] * synConstants[SYN_X2];
		synState[SYN_STATE_Y] = synState[SYN_STATE_X] * synConstants[SYN_Y1] + synState[SYN_STATE_Y] * synConstants[SYN_Y2];

		ftype gsyn = synState[SYN_STATE_Y] * synConstants[SYN_NORM];

		active[POS(synComp)] += (vmList[POS(synComp)] - synConstants[SYN_EK]) * gsyn;

		synState     += SYN_STATE_N;
		synConstants += SYN_CONST_N;
	}

	synState     -= SYN_STATE_N * synapseListSize;
	synConstants -= SYN_CONST_N * synapseListSize;
}


/**
 * Find the gate openings in the next time step
 * m(t + dt) = a + b m(t - dt)
 */
__device__ void evaluateGatesGNew( HinesStruct *hList, ftype *vmListLocal, int nChannels,
		ucomp *channelInfo, ftype *gatePar, ucomp *gateInfo, ftype *gateState) {

	HinesStruct & h = hList[blockIdx.x * blockDim.x + threadIdx.x];

	ftype alpha, beta, a, b;
	ftype V;
	ftype dtRev = 1/h.dt;

	int pos=0;
	for (int ch=0; ch<nChannels; ch++) {

		int nGates = channelInfo[ch*N_CHANNEL_FIELDS + CH_NGATES];
		//V = vmList[ channelInfo[ch*N_CHANNEL_FIELDS + CH_COMP] ];
		V = vmListLocal[ POS( channelInfo[ch*N_CHANNEL_FIELDS + CH_COMP] ) ];

		for (int gt=0; gt < nGates; gt++, pos++) {

            // (EXPONENTIAL): alpha(v) = A exp((v-V0)/B)
            // (SIGMOID):     alpha(v) = A / (exp((v-V0)/B) + 1)
            // (LINOID):      alpha(v) = A (v-V0) / (exp((v-V0)/B) - 1)

			// alpha_function
			ftype v0 = gatePar[A_V0];
			switch( gateInfo[pos * N_GATE_FIELDS + ALPHA_FUNCTION] ) {
			case EXPONENTIAL:
				alpha = gatePar[A_A] * expf((V-v0)/gatePar[A_B]);
				break;
			case SIGMOID:
				alpha = gatePar[A_A] / ( expf( (V-v0)/gatePar[A_B] ) + 1);
				break;
			case LINOID:
				alpha = (V != v0) ? gatePar[A_A] * (V-v0) / (expf((V-v0)/gatePar[A_B]) - 1) : gatePar[A_A] * gatePar[A_B];
				break;
			}

			// beta_function
			v0 = gatePar[B_V0];
			switch( gateInfo[pos * N_GATE_FIELDS + BETA_FUNCTION] ) {
			case EXPONENTIAL:
				beta = gatePar[B_A] * expf((V-v0)/gatePar[B_B]);
				break;
			case SIGMOID:
				beta = gatePar[B_A] / ( expf( (V-v0)/gatePar[B_B] ) + 1);
				break;
			case LINOID:
				beta = (V != v0) ? gatePar[B_A] * (V-v0) / (expf((V-v0)/gatePar[B_B]) - 1) : gatePar[B_A] * gatePar[B_B];
				break;
			}

			gatePar += N_GATE_FUNC_PAR;

			a = alpha / (dtRev  + (alpha + beta)/2);
			b = (dtRev - (alpha + beta)/2) / (dtRev + (alpha + beta)/2);

			gateState[POS(pos)] = a + b * gateState[POS(pos)];
		}

	}
}



__device__ void evaluateCurrentsGNew( HinesStruct *hList, ftype *activeList, ftype *vmListLocal,  int nChannels,
		ucomp *channelInfo, ucomp *gateInfo, ftype *gateState,
		int nComp, int compListSize, ucomp *compList, ftype *eLeak) {

	//ftype *Rm, ftype *active

	int neuron = blockIdx.x * blockDim.x + threadIdx.x;
	HinesStruct & h = hList[neuron];

	evaluateGatesGNew(hList, vmListLocal, nChannels, channelInfo, h.gatePar, gateInfo, gateState);

	ftype *channelEk = h.channelEk;
	ftype *channelGbar = h.channelGbar;

	for (int i=0; i<compListSize; i++)
		h.gActive[i] = 0;

	/**
	 * Update the channel conductances
	 */
	int pos = 0;
	int actCompPos = -1;
	int lastActComp = -1;

	for (int ch=0; ch<nChannels; ch++) {

		int nGates     = channelInfo[ch*N_CHANNEL_FIELDS + CH_NGATES];
		int comp       = channelInfo[ch*N_CHANNEL_FIELDS + CH_COMP];
		ftype gChannel = channelGbar[ch];


		for (int gt=0; gt < nGates; gt++, pos++) {

//			gateState[pos]=0;

			ftype state = gateState[POS(pos)];
			switch( gateInfo[pos * N_GATE_FIELDS + GATE_POWER] ) {
			case 4:
				gChannel *= (state*state*state*state);
				break;
			case 3:
				gChannel *= (state*state*state);
				break;
			case 2:
				gChannel *= (state*state);
				break;
			case 1:
				gChannel *= state;
				break;
			default:
				gChannel *= powf(state, gateInfo[pos * N_GATE_FIELDS + GATE_POWER] );
				break;
			}

		}

		activeList[ POS(comp) ] -= gChannel * channelEk[ch] ;

		if (comp != lastActComp) {
			actCompPos++;
			lastActComp = comp;
		}
		h.gActive[ actCompPos ] += gChannel;
	}

	for (int i=0; i<compListSize; i++) {
		unsigned int comp = compList[i];
		activeList[ POS(comp) ] -=  ( 1 / h.Rm[comp] ) * ( eLeak[i] );
	}
}


__device__ void upperTriangularizeAll(HinesStruct *hList, ftype *sTriangList,
				ftype *sLeftList, ucomp *sLeftListLine, ucomp *sLeftListColumn,
				ucomp *sLeftStartPos, ftype *rhsLocal, ftype *vmListLocal,

				int nChannels, ucomp *channelInfo, ucomp *gateInfo, ftype *gateState,
				int compListSize, ucomp *compList, ftype *eLeak,

				ftype *freeMem) {

	int neuron = blockIdx.x * blockDim.x + threadIdx.x;
	HinesStruct & h = hList[neuron];

	int nComp = h.nComp;
	int leftListSize = h.leftListSize;

	ftype *active = freeMem;
	freeMem = active + blockDim.x * nComp;

	ftype *Cm = h.Cm;
	ftype *curr = h.curr;


	for (int i=0; i<nComp; i++) {
		active[ POS(i)] = 0;
	}

	//__syncthreads();

	evaluateCurrentsGNew( hList, active, vmListLocal, nChannels,
			channelInfo, gateInfo, gateState,
			nComp, compListSize, compList, eLeak);

	evaluateSynapticCurrentsNew(hList, active, vmListLocal, h.currStep * h.dt,
			h.synapseListSize, h.synConstants, h.synState, h.synapseCompList, 	//
			h.activationListSize, h.activationList, h.activationListPos);

	ftype dtRec = 1/h.dt;
	//rhsLocal[POS(0)] = (-2) * vmListLocal[POS(0)] * Cm[0] * dtRec - curr[0] + active[POS(0)];
	for (int i=0; i<nComp; i++)
		rhsLocal[POS(i)] = (-2) * vmListLocal[POS(i)] * Cm[i] * dtRec - curr[i] + active[POS(i)];

	// ***
	// 1000ms 960 16 1 -> 0.125ms
	for (int k = 0; k < leftListSize; k++)
		sTriangList[k] = sLeftList[k];

	for (int i = 0; i < h.compListSize; i++) {

		int comp = h.compList[i];
		int pos = sLeftStartPos[ comp ];

		for (; sLeftListColumn[pos] < comp && pos < leftListSize ; pos++);

		sTriangList[pos] -= h.gActive[i];
	}


	// 1000ms 960 16 1 -> 0.640ms
	for (int k = 0; k < leftListSize; k++) {

		int c = sLeftListColumn[k];
		int l = sLeftListLine[k];

		if( c < l ) {

			int pos = sLeftStartPos[c];
			for (; c == sLeftListLine[pos]; pos++)
				if (sLeftListColumn[pos] == c)
					break;

			ftype mul = -sTriangList[k] / sTriangList[pos];

			pos = sLeftStartPos[c];
			int tempK = sLeftStartPos[l];

			for (; c == sLeftListLine[pos] && pos < leftListSize; pos++) {
				for (; sLeftListColumn[tempK] < sLeftListColumn[pos] && tempK < leftListSize ; tempK++);

				sTriangList[tempK] += sTriangList[pos] * mul;
			}
			rhsLocal[POS(l)] += rhsLocal[POS(c)] * mul;
		}
	}


}


__device__ void updateRhsG(HinesStruct *hList,
						   ftype *sMulList, ucomp *sMulListComp, ucomp *sMulListDest,
						   ftype *rhsLocal, ftype *vmListLocal,

						   int nChannels, ucomp *channelInfo, ucomp *gateInfo, ftype *gateState,
						   int compListSize, ucomp *compList, ftype *eLeak,

						   ftype *freeMem) {

	int neuron = blockIdx.x * blockDim.x + threadIdx.x;
	HinesStruct & h = hList[neuron];

	int nComp = h.nComp;

	ftype *active = freeMem;
	freeMem = active + blockDim.x * nComp;

	ftype *Cm = h.Cm;
	ftype *curr = h.curr;

	for (int i=0; i<nComp; i++) {
		active[ POS(i)] = 0;
	}


	//__syncthreads();

	evaluateCurrentsGNew( hList, active, vmListLocal, nChannels,
			channelInfo, gateInfo, gateState,
			nComp, compListSize, compList, eLeak);


	evaluateSynapticCurrentsNew(hList, active, vmListLocal, h.currStep * h.dt,
			h.synapseListSize, h.synConstants, h.synState, h.synapseCompList, 	//
			h.activationListSize, h.activationList, h.activationListPos);

	ftype dtRec = 1/h.dt;
	for (int i=0; i<nComp; i++)
		rhsLocal[POS(i)] = (-2) * vmListLocal[POS(i)] * Cm[i] * dtRec - curr[i] + active[POS(i)];

	int mulListSize = h.mulListSize;
	for (int mulListPos = 0; mulListPos < mulListSize; mulListPos++) {
		int dest = sMulListDest[mulListPos];
		int pos  = sMulListComp[mulListPos];
		rhsLocal[POS(dest)] += rhsLocal[POS(pos)] * sMulList[mulListPos];
	}

}

__device__ void backSubstituteG(HinesStruct *hList, 
								ftype *sTriangList, ucomp *sLeftListLine, ucomp *sLeftListColumn, 
								ftype *rhsLocal, ftype *vmListLocal, ftype* freeMem) {

	int neuron = blockIdx.x * blockDim.x + threadIdx.x;
	HinesStruct & h = hList[neuron];

	ucomp nComp = h.nComp;
	int leftListSize = h.leftListSize;

	ftype *vmTmpLocal = freeMem;

	if (h.triangAll == 0 && h.compListSize > 0) // has active channels only in soma
		vmTmpLocal[POS(nComp-1)] = rhsLocal[POS(nComp-1)] / ( sTriangList[(leftListSize-1)] - h.gActive[0]);
		//vmTmpLocal[POS(nComp-1)] = rhsLocal[POS(nComp-1)] / ( sTriangList[(leftListSize-1)] - h.gNaChannel[0] - h.gKChannel[0] );
	else
		vmTmpLocal[POS(nComp-1)] = rhsLocal[POS(nComp-1)] / sTriangList[(leftListSize-1)];


	ftype tmp = 0;
	for (int leftListPos = leftListSize-2; leftListPos >=0 ; leftListPos--) {
		ucomp line   = sLeftListLine[(leftListPos)];
		ucomp column = sLeftListColumn[(leftListPos)];
		if (line == column) {
			vmTmpLocal[POS(line)] = (rhsLocal[POS(line)] - tmp) * (1 / sTriangList[(leftListPos)]);
			tmp = 0;
		}
		else
			tmp += vmTmpLocal[POS(column)] * sTriangList[(leftListPos)];
	}

	for (int l = 0 ; l < nComp; l++)
		vmListLocal[POS(l)] = 2 * vmTmpLocal[POS(l)] - vmListLocal[POS(l)];

	//if (h.type == 0 && neuron == 1) printf("vmList=%.4f\n", vmListLocal[POS(0)]);

}

__global__ void solveMatrixG(HinesStruct *hList, int nSteps, int nNeurons, ftype *vmListGlobal) {

	int neuron = blockIdx.x * blockDim.x + threadIdx.x;
	if (neuron >= nNeurons) return;
	HinesStruct & h = hList[neuron];
	ucomp nComp = h.nComp;
	ucomp triangAll = h.triangAll;
	ucomp nGatesTotal = h.nGatesTotal;

	/******************************************************************************************
	 * Alocates the shared memory
	 *******************************************************************************************/

	// (ftype * 5 + ucomp * 10) * nComp
	// ftype=4 e ucomp=2 e ncomp = 8   ->  320 bytes 
	// ftype=4 e ucomp=2 e ncomp = 64  -> 2560 bytes  
	extern __shared__ ftype sharedMem[]; 
	ftype *sLeftList       = (ftype *)sharedMem;
	ucomp *sLeftListLine   = (ucomp *)&(sLeftList[h.leftListSize]); 	
	ucomp *sLeftListColumn = (ucomp *)&(sLeftListLine[h.leftListSize]); 

	ftype *sMulList     = (ftype *)&(sLeftListColumn[h.leftListSize]); // mulSize is zero when triangAll is 1
	ucomp *sMulListComp = (ucomp *)&(sMulList[h.mulListSize]); 	
	ucomp *sMulListDest = (ucomp *)&(sMulListComp[h.mulListSize]); 

	ucomp *sLeftStartPos = (ucomp *)&(sMulListDest[h.mulListSize]);

	ucomp *sActiveCompList = &(sLeftStartPos[nComp]); // No significant speedup
	ucomp *sChannelInfo    = &(sActiveCompList[h.compListSize]); // No significant speedup
	ucomp *sGateInfo       = &(sChannelInfo[h.nChannels * N_CHANNEL_FIELDS]); // small speedup

//	int nChannelTypes = h.nChannelTypes;
//	ftype *sTau		= (ftype *)&(sGateInfo[nGatesTotal * N_GATE_FIELDS]);
//	ftype *sGmax 	= (ftype *)&(sTau[nChannelTypes*2]);
//	ftype *sEsyn 	= (ftype *)&(sGmax[nChannelTypes]);

	ftype *lastSharedAddress = (ftype *)&(sGateInfo[nGatesTotal * N_GATE_FIELDS]);

	/******************************************************************************************
	 * Allocate for each individual neuron
	 *******************************************************************************************/

	// nThreads * nComp * ftype * 2
	// 32 * [8 ] * 4 * 2 = 32 * 64 = 2K
	// 32 * [32] * 4 * 2 =         = 8K

	ftype *rhsLocal = (ftype *)lastSharedAddress;
	ftype *vmListLocal = rhsLocal + blockDim.x * nComp;

//	ftype *sChannelEk   = vmListLocal  + blockDim.x * nComp;
//	ftype *sChannelGbar = sChannelEk   + blockDim.x * h.nChannels;
//	ftype *sELeak       = sChannelGbar + blockDim.x * h.nChannels;
//	ftype *sGatePar     = sGateState   + blockDim.x * (h.gatePar - h.gateState);

	ftype *sGateState   = vmListLocal + blockDim.x * nComp; //sELeak + blockDim.x * h.compListSize;
	ftype *freeMem = sGateState + blockDim.x * nGatesTotal;
	ftype *sTriangList = 0;

	if (triangAll == 1) {
		sTriangList = freeMem + threadIdx.x * h.leftListSize;
		freeMem  = freeMem + blockDim.x * h.leftListSize;
	}

	/******************************************************************************************
	 * Initializaes the shared memory
	 *******************************************************************************************/

//	for (int id=0; id < nChannelTypes; id++ ) {
//		sTau[2*id]   = h.tau[2*id];
//		sTau[2*id+1] = h.tau[2*id+1];
//		sGmax[id] 	 = h.gmax[id];
//		sEsyn[id] 	 = h.esyn[id];
//
//	}

	for (int k=0; k < nComp; k ++ )
		sLeftStartPos[k] = h.leftStartPos[k];

	for (int k=0; k < nGatesTotal; k ++ )
		sGateState[POS(k)] = h.gateState[k];

	for (int i=nGatesTotal*N_GATE_FIELDS-1; i >=0; i--)
		sGateInfo[i] = h.gateInfo[i];

	for (int i=h.nChannels*N_CHANNEL_FIELDS-1; i >= 0 ; i--)
		sChannelInfo[i] = h.channelInfo[i];

	for (int i=h.compListSize-1; i >= 0 ; i--)
		sActiveCompList[i] = h.compList[i];

	for (int k=0; k < h.leftListSize; k ++ ) {
		if (triangAll == 0) sLeftList[k] = h.triangList[k];
		else				sLeftList[k] = h.leftList[k];
		sLeftListLine[k]   = h.leftListLine[k];
		sLeftListColumn[k] = h.leftListColumn[k];
	}

	if (triangAll == 0) {
		for (int k=0; k < h.mulListSize; k ++ ) {
			sMulList[k]     = h.mulList[k];
			sMulListComp[k] = h.mulListComp[k];
			sMulListDest[k] = h.mulListDest[k];
		}
	}

	for (int k=0; k < nComp; k++ )
		vmListLocal[POS(k)] = h.vmList[k];

//	for (int k=0; k < h.compListSize; k++ ) {
//		nGate[POS(k)] = h.n[k];
//		hGate[POS(k)] = h.h[k];
//		mGate[POS(k)] = h.m[k];
//	}

	/******************************************************************************************
	 * Perform the simulation
	 *******************************************************************************************/

	ftype dt = h.dt;
	int currStep = h.currStep;
	ucomp nGeneratedSpikes = 0;

	for(int gStep = 0; gStep < nSteps; gStep++ ) {

		if (triangAll == 0) {
			updateRhsG(hList, sMulList, sMulListComp, sMulListDest, rhsLocal, vmListLocal,
					h.nChannels, sChannelInfo, sGateInfo, sGateState,
					h.compListSize, sActiveCompList, h.eLeak, freeMem); // RYC
			backSubstituteG(hList, sLeftList, sLeftListLine, sLeftListColumn, rhsLocal, vmListLocal, freeMem); // RYC
		}
		else {

			upperTriangularizeAll(hList, sTriangList, sLeftList, sLeftListLine, sLeftListColumn,
					sLeftStartPos, rhsLocal, vmListLocal,
					   h.nChannels, sChannelInfo,
					   sGateInfo, sGateState,
					   h.compListSize, sActiveCompList, h.eLeak,
					   freeMem);
			backSubstituteG(hList, sTriangList, sLeftListLine, sLeftListColumn, rhsLocal, vmListLocal, freeMem); // RYC
		}
		//printf ("SolveMatrixG: Ok2\n");

		for (int k=0; k<nComp; k++) {
			int index = k * nSteps + gStep;
			h.vmTimeSerie[index] = vmListLocal[POS(k)]; // RYC
		}

		currStep = currStep + 1;


		if (vmListLocal[POS(nComp-1)] >= h.threshold && ((currStep * dt) - h.lastSpike) > h.minSpikeInterval) {

			h.spikeTimes[nGeneratedSpikes] = currStep * dt;
			h.lastSpike = currStep * dt;
			nGeneratedSpikes++;
		}

		h.currStep = currStep;
	}

	h.nGeneratedSpikes[neuron] = nGeneratedSpikes;

	//__syncthreads();

	for (int k=0; k<nComp; k++) {
		h.rhsM[k] = rhsLocal[POS(k)];
		h.vmList[k] = vmListLocal[POS(k)]; // RYC
	}

	for (int k=0; k < nGatesTotal; k ++ )
		h.gateState[k] = sGateState[POS(k)];

//	for (int k=0; k < h.compListSize; k++ ) {
//		h.n[k] = nGate[POS(k)];
//		h.h[k] = hGate[POS(k)];
//		h.m[k] = mGate[POS(k)];
//	}

	vmListGlobal[neuron] = vmListLocal[POS(nComp-1)];

	// used only for debugging
	//h.active[0] = ((char *)freeMem - (char *)sharedMem)*1.0;

}
