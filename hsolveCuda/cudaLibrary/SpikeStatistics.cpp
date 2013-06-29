/*
 * SpikeStatistics.cpp
 *
 *  Created on: 10/09/2009
 *      Author: rcamargo
 */

#include "SpikeStatistics.hpp"
#include "HinesMatrix.hpp"

#include <cstdio>

SpikeStatistics::SpikeStatistics(int *nNeurons, int nNeuronTypes, int totalTypes, int *typeList) {

    nSpkfile = 0;
    lastSpkfile = 0;

	this->typeList     = typeList;
	this->totalTypes   = totalTypes;
	this->nNeurons     = nNeurons;
	this->nNeuronTypes = nNeuronTypes;

	totalGeneratedSpikes = new ftype *[totalTypes];
	totalReceivedSpikes  = new ftype *[totalTypes];
	lastGeneratedSpikeTimes = new ftype *[totalTypes];

	for (int type=0; type<totalTypes; type++) {
		totalGeneratedSpikes[type] = new ftype[nNeurons[type]];
		totalReceivedSpikes[type]  = new ftype[nNeurons[type]];
		lastGeneratedSpikeTimes[type] = new ftype[nNeurons[type]];

		for (int neuron=0; neuron < nNeurons[type]; neuron++) {
			totalGeneratedSpikes[type][neuron] = 0;
			totalReceivedSpikes[type][neuron]  = 0;
			lastGeneratedSpikeTimes[type][neuron] = 0;
		}
	}

	totalNeurons = 0;

	bench.meanGenSpikesType = new ftype[nNeuronTypes];
	bench.meanRecSpikesType = new ftype[nNeuronTypes];
	this->totalNeuronsType  = new int[nNeuronTypes];
	for (int neuronType=0; neuronType<this->nNeuronTypes; neuronType++) {
		totalNeuronsType[neuronType]    = 0;
		bench.meanGenSpikesType[neuronType] = 0;
		bench.meanRecSpikesType[neuronType] = 0;
	}

	for (int groupType=0; groupType<totalTypes; groupType++) {
		totalNeurons += nNeurons[groupType];
		totalNeuronsType[ typeList[groupType] ] += nNeurons[ groupType ];
	}

}

SpikeStatistics::~SpikeStatistics() {

	if (nSpkfile != 0) fclose(nSpkfile);
	if (lastSpkfile != 0) fclose(lastSpkfile);
}

void SpikeStatistics::addGeneratedSpikes(int type, int neuron, ftype *spikeTimes, int nSpikes) {
	totalGeneratedSpikes[type][neuron] += nSpikes;
	if (spikeTimes != NULL)
		lastGeneratedSpikeTimes[type][neuron] = spikeTimes[nSpikes-1];
	else
		lastGeneratedSpikeTimes[type][neuron] = 0.00;
}

void SpikeStatistics::addReceivedSpikes(int type, int neuron, int nReceivedSpikes) {
	totalReceivedSpikes[type][neuron] += nReceivedSpikes;
}

void SpikeStatistics::printKernelSpikeStatistics( ftype currentTime) {

    if (nSpkfile == 0)
    	nSpkfile = fopen("nSpikeKernel.dat", "w");

    if (lastSpkfile == 0)
    	lastSpkfile = fopen("lastSpikeKernel.dat", "w");

	for (int type=0; type<totalTypes; type++) {
		fprintf(nSpkfile,	"%-10.2f\ttype=%d | ", currentTime, type);
		fprintf(lastSpkfile,"%-10.2f\ttype=%d | ", currentTime, type);

		for (int neuron=0; neuron < nNeurons[type]; neuron++) {

			fprintf(nSpkfile, 	 "%10.1f ", totalGeneratedSpikes[type][neuron]);
			fprintf(lastSpkfile, "%10.2f ", lastGeneratedSpikeTimes[type][neuron]);
		}

		fprintf(nSpkfile, 	 "\n");
		fprintf(lastSpkfile, "\n");
	}

	fprintf(nSpkfile, 	 "\n");
	fprintf(lastSpkfile, "\n");
}

void SpikeStatistics::printSpikeStatistics(const char *filename, ftype currentTime, BenchTimes & bench) {

//	ftype genSpikes = 0;
//	ftype recSpikes = 0;


	FILE *outFile = fopen(filename, "w");
	fprintf(outFile, "# totalTime=%f, totalNeurons=%d, nTypes=%d\n", currentTime, totalNeurons, totalTypes);

	for (int type=0; type<totalTypes; type++) {

		for (int neuron=0; neuron < nNeurons[type]; neuron++) {

			bench.meanGenSpikes += totalGeneratedSpikes[type][neuron];
			bench.meanRecSpikes += totalReceivedSpikes[type][neuron];

			fprintf(outFile, "[%2d][%6d]\t%.1f\t%.1f\t%10.2f\n",
					type, neuron, totalGeneratedSpikes[type][neuron],
					totalReceivedSpikes[type][neuron], lastGeneratedSpikeTimes[type][neuron]);
		}

		for (int neuron=0; neuron < nNeurons[type]; neuron++) {
			bench.meanGenSpikesType[typeList[type]] += totalGeneratedSpikes[type][neuron];
			bench.meanRecSpikesType[typeList[type]] += totalReceivedSpikes[type][neuron];
		}

	}

	bench.meanGenSpikes /= totalNeurons;
	bench.meanRecSpikes /= totalNeurons;

	for (int neuronType=0; neuronType < this->nNeuronTypes; neuronType++) {
		bench.meanGenSpikesType[neuronType] /= totalNeuronsType[neuronType];
		bench.meanRecSpikesType[neuronType] /= totalNeuronsType[neuronType];
	}

	if (nNeuronTypes == 3)
		printf("meanGenSpikes[T|P|I|B]=[%3.2f|%3.2f|%3.2f|%3.2f]\nmeanRecSpikes[T|P|I|B]=[%5.2f|%5.2f|%5.2f|%5.2f]\n",
				bench.meanGenSpikes, bench.meanGenSpikesType[PYRAMIDAL_CELL], bench.meanGenSpikesType[INHIBITORY_CELL], bench.meanGenSpikesType[BASKET_CELL],
				bench.meanRecSpikes, bench.meanRecSpikesType[PYRAMIDAL_CELL], bench.meanRecSpikesType[INHIBITORY_CELL], bench.meanRecSpikesType[BASKET_CELL]);
	else if (nNeuronTypes == 2)
		printf("meanGenSpikes[T|P|I|B]=[%3.2f|%3.2f|%3.2f]\nmeanRecSpikes[T|P|I|B]=[%5.2f|%5.2f|%5.2f]\n",
				bench.meanGenSpikes, bench.meanGenSpikesType[PYRAMIDAL_CELL], bench.meanGenSpikesType[INHIBITORY_CELL],
				bench.meanRecSpikes, bench.meanRecSpikesType[PYRAMIDAL_CELL], bench.meanRecSpikesType[INHIBITORY_CELL]);

	fclose(outFile);
}

