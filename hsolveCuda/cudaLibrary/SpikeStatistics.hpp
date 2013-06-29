/*
 * SpikeStatistics.hpp
 *
 *  Created on: 10/09/2009
 *      Author: rcamargo
 */

#ifndef SPIKESTATISTICS_HPP_
#define SPIKESTATISTICS_HPP_

#include "PlatformFunctions.hpp"
#include <cstdio>

class SpikeStatistics {

    FILE *nSpkfile;
    FILE *lastSpkfile;

	int *nNeurons;
	int *typeList;
	int totalTypes;

	int totalNeurons;

	int nNeuronTypes;
	int *totalNeuronsType;

	ftype **totalGeneratedSpikes;
	ftype **totalReceivedSpikes;

	ftype **lastGeneratedSpikeTimes;

public:
	SpikeStatistics(int *nNeurons, int nNeuronTypes, int totalTypes, int *typeList);
	virtual ~SpikeStatistics();

	void addGeneratedSpikes(int type, int neuron, ftype *spikeTimes, int nSpikes);

	void addReceivedSpikes(int type, int neuron, int nReceivedSpikes);

	void printSpikeStatistics(const char *filename, ftype currentTime, BenchTimes & bench);

	void printKernelSpikeStatistics(ftype currentTime);
};

#endif /* SPIKESTATISTICS_HPP_ */
