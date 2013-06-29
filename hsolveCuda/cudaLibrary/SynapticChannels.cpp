/*
 * SynapticChannels.cpp
 *
 *  Created on: 07/08/2009
 *      Author: rcamargo
 */

#include "SynapticChannels.hpp"
#include <cstdio>
#include <cassert>
#include <cmath>

SynapticChannels::SynapticChannels(ftype *synapticCurrent, ftype *vmList, int nComp) {
	this->synapticCurrent = synapticCurrent;
	this->vmList = vmList;
	this->nAddedSpikes = 0;
	spikeList = 0;
	spikeListSize = 0;
	synapseWeightList = 0;
	activationList = 0;
	activationListSize = 0;

	pthread_mutex_init( &addSpikeMutex, NULL );
	createChannelsAndSynapses(nComp);
}

// TODO: check memory free
SynapticChannels::~SynapticChannels() {
	delete[] tau;

	if (synapseCompList != 0) delete[] synapseCompList;
	if (spikeList != 0) delete[] spikeList;
	//if (synapseWeightList != 0) delete[] synapseWeightList;

	if (activationList != 0) delete[] activationList;



}

void SynapticChannels::createChannelsAndSynapses(int nComp) {

	nChannelTypes = 2;
	tau  = new ftype[4*nChannelTypes];
	gmax =  tau + 2*nChannelTypes;//new ftype[nChannelTypes];
	esyn = gmax +   nChannelTypes;//new ftype[nChannelTypes];

	/**
	 * Creates AMPA channel
	 */
	tau[2*SYNAPSE_AMPA] = 4;
	tau[2*SYNAPSE_AMPA+1] = 4;
	gmax[SYNAPSE_AMPA] = 20 * 500e-9;
	esyn[SYNAPSE_AMPA] = 70;

	tau[2*SYNAPSE_GABA] = 4;//17;
	tau[2*SYNAPSE_GABA+1] = 4;//100;
	gmax[SYNAPSE_GABA] = 20 * 1250e-9;
	esyn[SYNAPSE_GABA] = -75;

	/**
	 * Creates synapses
	 */
	synapseListSize = 2;
	nDelieveredSpikes = new int[synapseListSize];

	synapseCompList = new ucomp[4*synapseListSize];
	synapseTypeList = synapseCompList + synapseListSize;
	synSpikeListPos = synapseTypeList + synapseListSize;
	synSpikeListTmp = synSpikeListPos + synapseListSize;

	for (int i=0; i<synapseListSize; i++) {
		synSpikeListPos[i] = 0;
		synSpikeListTmp[i] = 0;
	}

	synapseCompList[0] = 0;
	synapseTypeList[0] = SYNAPSE_AMPA;

	synapseCompList[1] = nComp-1;
	synapseTypeList[1] = SYNAPSE_GABA;

	synSpikeSet.resize(2);
//	synSpikeMap[0] = new SpikeMap();
//	synSpikeMap[1] = new SpikeMap();
//	synSpikeSet.push_back( new SpikeSet() );
//	synSpikeSet.push_back( new SpikeSet() );

}

ftype SynapticChannels::getCurrent(ucomp synType, ftype spikeTime, ftype weight, ucomp comp, ftype currTime) {

	//printf ("weight=%f\n", weight);
	ftype gsyn = 0;
	ftype current = 0;
	if (synType == SYNAPSE_AMPA) {
		ftype r = (currTime - spikeTime) / tau[2*SYNAPSE_AMPA];
		gsyn = gmax[SYNAPSE_AMPA] * r * exp(1 - r) * weight;
		current = (vmList[comp] - esyn[SYNAPSE_AMPA]) * gsyn;
	}
	else if (synType == SYNAPSE_GABA) {
		ftype r = (currTime - spikeTime) / tau[2*SYNAPSE_GABA];
		gsyn = gmax[SYNAPSE_GABA] * r * exp(1 - r) * weight;
		current = (vmList[comp] - esyn[SYNAPSE_GABA]) * gsyn;
	}
	else
		printf ("ERROR: SynapticChannels::getCurrent -> Defined synapse type not found.\n");

	return current;
}

void SynapticChannels::configureSynapticActivationList(ftype dt, int listSize) {

	if (activationList != 0)
		delete[] activationList;

	synapseListSize = 2;
	activationListSize = listSize;

	synapseCompList = new ucomp[2*synapseListSize];
	synapseTypeList = synapseCompList + synapseListSize;
	synapseCompList[0] = 0;
	synapseCompList[1] = 3; // TODO: should be ncomp-1
	synapseTypeList[0] = SYNAPSE_AMPA;
	synapseTypeList[1] = SYNAPSE_GABA;

	activationListPos    = new ucomp[synapseListSize];
	activationListPos[0] = 0;
	activationListPos[1] = 0;

	activationList = new ftype[synapseListSize * activationListSize];
	for (int syn=0; syn < synapseListSize; syn++)
		for (int i=0; i < activationListSize; i++)
			activationList[syn * activationListSize + i] = 0;

	synState = new ftype[synapseListSize * SYN_STATE_N];
	for (int syn=0; syn < synapseListSize; syn++) {
		synState[SYN_STATE_X] = 0;
		synState[SYN_STATE_Y] = 0;

		synState += SYN_STATE_N;
	}

	// TODO: used only for testing
	synCurrentTmp = new ftype[synapseListSize * synapseListSize];

	synConstants = new ftype[synapseListSize * SYN_CONST_N];
	for (int syn=0; syn < synapseListSize; syn++) {
		ftype tau1 = tau[2*syn];   // tau1
		ftype tau2 = tau[2*syn+1]; // tau2

		synConstants[SYN_X1] = tau1 * ( 1.0 - expf( -dt / tau1 ) );
		synConstants[SYN_X2] = expf( -dt / tau1 );

		synConstants[SYN_Y1] = tau2 * ( 1.0 - expf( -dt / tau2 ) );
		synConstants[SYN_Y2] = expf( -dt / tau2 );

		synConstants[SYN_EK]  = esyn[syn];
		synConstants[SYN_MOD] = 1;

		if (tau1 == tau2)
			synConstants[SYN_NORM] = gmax[syn] * expf(1.0) / tau1; // TODO: 1?
		else {
			ftype tpeak = tau1 * tau2 * logf( tau1/tau2 ) / ( tau1-tau2 );
			synConstants[SYN_NORM] =
					gmax[syn] * ( tau1-tau2 ) /	( tau1 * tau2 * ( expf( -tpeak / tau1 ) - expf( -tpeak / tau2 ) ) );
		}

		synConstants += SYN_CONST_N;
	}

	synState     -= SYN_STATE_N * synapseListSize;
	synConstants -= SYN_CONST_N * synapseListSize;
}

void SynapticChannels::clearSynapticActivationList() {

	for (int syn=0; syn < synapseListSize; syn++)
		for (int i=0; i < activationListSize; i++)
			activationList[syn * activationListSize + i] = 0;
}


int SynapticChannels::getAndResetNumberOfAddedSpikes() {
	int tmp = nAddedSpikes;
	nAddedSpikes = 0;
	return tmp;
}

void SynapticChannels::evaluateCurrentsNew(ftype currTime) {

	for (int syn=0; syn < synapseListSize; syn++) {
		int currPos = (syn * activationListSize) + activationListPos[syn];
		ftype activation = activationList[ currPos ];
		activationList[ currPos ] = 0;

		activationListPos[syn] = (activationListPos[syn] + 1) % activationListSize;

		int synComp = synapseCompList[syn];

		synState[SYN_STATE_X] = synConstants[SYN_MOD] * activation * synConstants[SYN_X1] + synState[SYN_STATE_X] * synConstants[SYN_X2];
		synState[SYN_STATE_Y] = synState[SYN_STATE_X] * synConstants[SYN_Y1] + synState[SYN_STATE_Y] * synConstants[SYN_Y2];

		ftype gsyn = synState[SYN_STATE_Y] * synConstants[SYN_NORM];

		synapticCurrent[synComp] += (vmList[synComp] - synConstants[SYN_EK]) * gsyn;
		//synCurrentTmp[synapseListSize + syn] += (vmList[synComp] - synConstants[SYN_EK]) * gsyn; // Used only for testing

		synState     += SYN_STATE_N;
		synConstants += SYN_CONST_N;
	}

	synState     -= SYN_STATE_N * synapseListSize;
	synConstants -= SYN_CONST_N * synapseListSize;
}


