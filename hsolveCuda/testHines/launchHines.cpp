/*
 * launchHines.cpp
 *
 *  Created on: 02/06/2009
 *      Author: Raphael Y. de Camargo
 *      Affiliation: Universidade Federal do ABC (UFABC), Brazil
 */

/**
 * - Create matrix for linear Cable [Ter] OK
 * - Solve matrix with Gaussian elimination [Ter] OK
 * - Print Vm in an output file (input for Gnuplot) [Ter] OK
 * - Add current injection and leakage [Qua] OK
 * - Check constants and compare with GENESIS [Qua] OK
 * - Add soma [Qua] OK
 * - Add branched Tree [Qua] OK
 * - Support for multiples neurons at the same time OK
 * - Allocate all the memory in sequence OK
 * - Simulation in the GPU as a series of steps OK
 * - Larger neurons OK
 *
 * - Active Channels on CPUs [Ter-16/06] OK
 * - Active Channels on GPUs [Qui-18/06] OK
 *
 * - Usage of shared tables for active currents (For GPUs may be not useful) (1) [Seg-23/06]
 * - Optimize performance for larger neurons (GPU shared memory) (2) [Seg-23/06]
 * - Optimize performance for larger neurons (Memory Coalescing) (3) [Ter-24/06]
 * 		- Each thread reads data for multiple neurons
 *
 * - Optimizations (Useful only for large neurons)
 *   - mulList and leftMatrix can be represented as lists (use macros)
 *     - Sharing of matrix values among neurons (CPU {cloneNeuron()} e GPU)
 *     - mulList e tmpVmList can share the same matrix (better to use lists)
 *
 * - Support for communication between neurons [Future Work]
 * - Support for multiple threads for each neuron (from different blocks, y axis) [Future Work]
 */

#include <cstdio>
#include <cassert>
#include <cstdlib>
#include <ctime>
#include <pthread.h>
#include <unistd.h>
#include <cmath>

#include "../cudaLibrary/SharedNeuronGpuData.hpp"
#include "../cudaLibrary/KernelInfo.hpp"
#include "../cudaLibrary/ThreadInfo.hpp"
#include "../cudaLibrary/PerformSimulation.hpp"
#include "../cudaLibrary/Connections.hpp"
#include "../cudaLibrary/HinesMatrix.hpp"
#include "../cudaLibrary/ActiveChannels.hpp"
#include "../cudaLibrary/PlatformFunctions.hpp"
#include "../cudaLibrary/SpikeStatistics.hpp"

using namespace std;

void *launchHostExecution(void *ptr) {

	ThreadInfo *tInfo = (ThreadInfo *)ptr;

	/**
	 * Launches the execution of all threads
	 */
	PerformSimulation *simulation = new PerformSimulation(tInfo);
	simulation->launchExecution();
	//simulation->performHostExecution();

	return 0;
}


void *launchDeviceExecution(void *threadInfo) {

	ThreadInfo *tInfo = (ThreadInfo *)threadInfo;

	/**
	 * Launches the execution of all threads
	 */
	PerformSimulation *simulation = new PerformSimulation(tInfo);
	simulation->launchExecution();

	return 0;
}


ThreadInfo *createInfoArray(int nThreads, ThreadInfo *model){
	ThreadInfo *tInfoArray = new ThreadInfo[nThreads];
	for (int i=0; i<nThreads; i++) {
		tInfoArray[i].sharedData 	= model->sharedData;
		tInfoArray[i].nComp			= model->nComp;
		tInfoArray[i].nNeurons		= model->nNeurons;
		tInfoArray[i].nNeuronsTotalType = model->nNeuronsTotalType;

		tInfoArray[i].nTypes		= model->nTypes;
		tInfoArray[i].totalTypes	= model->totalTypes;
	}

	return tInfoArray;
}

void configureNeuronTypes(char*& simType, ThreadInfo*& tInfo, int nNeuronsTotal,  char *configFileName) {

	int nComp = 4;

	tInfo->nTypes = 3;
	tInfo->totalTypes = tInfo->nTypes * tInfo->sharedData->nThreadsCpu;

	tInfo->nNeurons = new int[tInfo->totalTypes];
	tInfo->nComp = new int[tInfo->totalTypes];
	tInfo->sharedData->typeList = new int[tInfo->totalTypes];
	tInfo->nNeuronsTotalType = new int[tInfo->nTypes];
	for (int type=0; type < tInfo->nTypes; type++)
		tInfo->nNeuronsTotalType[ type ] = 0;

	tInfo->sharedData->matrixList = new HinesMatrix *[tInfo->totalTypes];
	for (int i = 0; i < tInfo->totalTypes; i += tInfo->nTypes) {
		tInfo->nNeurons[i] = nNeuronsTotal / (tInfo->totalTypes);
		tInfo->nComp[i] = nComp;
		tInfo->sharedData->typeList[i] = PYRAMIDAL_CELL;
		tInfo->nNeuronsTotalType[ tInfo->sharedData->typeList[i] ] += tInfo->nNeurons[i];

		tInfo->nNeurons[i + 1] = nNeuronsTotal / (tInfo->totalTypes);
		tInfo->nComp[i + 1] = nComp;
		tInfo->sharedData->typeList[i + 1] = INHIBITORY_CELL;
		tInfo->nNeuronsTotalType[ tInfo->sharedData->typeList[i] ] += tInfo->nNeurons[i];

		tInfo->nNeurons[i + 2] = nNeuronsTotal / (tInfo->totalTypes);
		tInfo->nComp[i + 2] = nComp;
		tInfo->sharedData->typeList[i + 2] = BASKET_CELL;
		tInfo->nNeuronsTotalType[ tInfo->sharedData->typeList[i] ] += tInfo->nNeurons[i];

	}
}

void configureSimulation(char *simType, ThreadInfo *& tInfo, int nNeurons, char mode, char *configFile)
{

	// Configure the types and number of neurons
	configureNeuronTypes(simType, tInfo, nNeurons, configFile);

	// defines some default values
	tInfo->sharedData->inputSpikeRate = 0.1;
	tInfo->sharedData->pyrPyrConnRatio   = 0.1;
	tInfo->sharedData->pyrInhConnRatio   = 0.1;
	tInfo->sharedData->totalTime   = 100; // in ms

	tInfo->sharedData->randWeight = 1;

	if (simType[0] == 'n' || simType[0] == 'd') {
		printf ("Simulation configured as: Running scalability experiments.\n");

		benchConf.printSampleVms = 1;
		benchConf.printAllVmKernelFinish = 0;
		benchConf.printAllSpikeTimes = 0;
		benchConf.checkGpuComm = 0;

		if (mode=='G')      benchConf.setMode(NN_GPU, NN_GPU);
		else if (mode=='H') benchConf.setMode(NN_GPU, NN_CPU);
		else if (mode=='C') benchConf.setMode(NN_CPU, NN_CPU);
		else if (mode=='T') benchConf.setMode(NN_GPU, NN_TEST);

		if (simType[0] == 'n') benchConf.gpuCommBenchMode = GPU_COMM_SIMPLE;
		else if (simType[0] == 'd') benchConf.gpuCommBenchMode = GPU_COMM_DETAILED;

		tInfo->sharedData->totalTime   = 1000;
		tInfo->sharedData->inputSpikeRate = 0.01;
		tInfo->sharedData->connectivityType = CONNECT_RANDOM_1;

		tInfo->sharedData->excWeight = 0.01;  //1.0/(nPyramidal/100.0); 0.05
		tInfo->sharedData->pyrInhWeight = 0.1; //1.0/(nPyramidal/100.0);
		tInfo->sharedData->inhPyrWeight = 1;

		if (simType[1] == '0') { // 200k: 0.79
			tInfo->sharedData->pyrPyrConnRatio   = 0; // nPyramidal
			tInfo->sharedData->pyrInhConnRatio   = 0; // nPyramidal
			tInfo->sharedData->inputSpikeRate = 0.02; // increases the input
		}
		else if (simType[1] == '1') {
			tInfo->sharedData->pyrPyrConnRatio   = 100.0 / (nNeurons/tInfo->nTypes); // nPyramidal //100
			tInfo->sharedData->pyrInhConnRatio   = 100.0 / (nNeurons/tInfo->nTypes); // nPyramidal //100

			if (simType[2] == 'l') { // 200k: 0.92
				tInfo->sharedData->excWeight    = 0.030;
				tInfo->sharedData->pyrInhWeight = 0.035;
				tInfo->sharedData->inhPyrWeight = 10;
			}
			if (simType[2] == 'm') { // 200k: 1.93
				tInfo->sharedData->excWeight    = 0.045;
				tInfo->sharedData->pyrInhWeight = 0.020;
				tInfo->sharedData->inhPyrWeight = 4;
			}
			if (simType[2] == 'h') { // 200k: 5.00
				tInfo->sharedData->excWeight    = 0.100;
				tInfo->sharedData->pyrInhWeight = 0.030;
				tInfo->sharedData->inhPyrWeight = 1;
			}
		}
		else if (simType[1] == '2') {
			tInfo->sharedData->pyrPyrConnRatio   = 1000.0 / (nNeurons/tInfo->nTypes); // nPyramidal
			tInfo->sharedData->pyrInhConnRatio   = 1000.0 / (nNeurons/tInfo->nTypes); // nPyramidal

			if (simType[2] == 'l') { // 100k: 1.19
				tInfo->sharedData->excWeight    = 0.004;
				tInfo->sharedData->pyrInhWeight = 0.004;
				tInfo->sharedData->inhPyrWeight = 10;
			}
			if (simType[2] == 'm') { // 10k: 2.04
				tInfo->sharedData->excWeight    = 0.005;
				tInfo->sharedData->pyrInhWeight = 0.003;
				tInfo->sharedData->inhPyrWeight = 4;
			}
			if (simType[2] == 'h') { // 10k: 5.26
				tInfo->sharedData->excWeight    = 0.008;
				tInfo->sharedData->pyrInhWeight = 0.004;
				tInfo->sharedData->inhPyrWeight = 1;
			}

		}
	}


}

// uA, kOhm, mV, cm, uF
int main(int argc, char **argv) {

	int nProcesses = 1;
    int currentProcess=0;

	bench.start = gettimeInMilli();

	ThreadInfo *tInfo = new ThreadInfo;
	tInfo->sharedData = new SharedNeuronGpuData;
    tInfo->sharedData->kernelInfo = new KernelInfo;

	tInfo->sharedData->nBarrier = 0;
	tInfo->sharedData->mutex = new pthread_mutex_t;
	tInfo->sharedData->cond = new pthread_cond_t;
	pthread_cond_init (  tInfo->sharedData->cond, NULL );
	pthread_mutex_init( tInfo->sharedData->mutex, NULL );

	tInfo->sharedData->synData = 0;
	tInfo->sharedData->hGpu = 0;
	tInfo->sharedData->hList = 0;
	tInfo->sharedData->globalSeed = time(NULL);

	benchConf.assertResultsAll = 1; // TODO: was 1
	benchConf.printSampleVms = 0;
	benchConf.printAllVmKernelFinish = 0;
	benchConf.printAllSpikeTimes = 1;
	benchConf.verbose = 0;

	int nNeuronsTotal = 0;

	if ( argc < 4 ) {
		printf("Invalid arguments!\n Usage: %s <mode> <simType> <nNeurons> <nGPUs> [seed]\n", argv[0]);
		printf("Invalid arguments!\n Usage: %s <mode> <simType> <nNeurons> <configFile> [seed]\n", argv[0]);

		exit(-1);
	}

	char mode = argv[1][0];
	assert (mode == 'C' || mode == 'G' || mode == 'H' || mode == 'B' || mode == 'T');

    char *simType = argv[2];

	nNeuronsTotal = atoi(argv[3]);
	assert ( 0 < nNeuronsTotal && nNeuronsTotal < 4096*4096);

	tInfo->sharedData->nThreadsCpu = atoi(argv[4]);
	configureSimulation(simType, tInfo, nNeuronsTotal, mode, 0);


	if (argc > 5)
		tInfo->sharedData->globalSeed = atoi(argv[4])*123;

	int nThreadsCpu = tInfo->sharedData->nThreadsCpu;

	// Configure the simulationSteps
	tInfo->sharedData->randBuf = new random_data *[nThreadsCpu];

	pthread_t *thread1 = new pthread_t[nThreadsCpu];
	ThreadInfo *tInfoArray = createInfoArray(nThreadsCpu, tInfo);
	for (int t=0; t<nThreadsCpu; t++) {

			if (mode == 'C' || mode == 'B')
				pthread_create ( &thread1[t], NULL, launchHostExecution, &(tInfoArray[t]));

			if (mode == 'G' || mode == 'H' || mode == 'B' || mode == 'T')
				pthread_create ( &thread1[t], NULL, launchDeviceExecution, &(tInfoArray[t]));

			//pthread_detach(thread1[t]);
	}

	for (int t=0; t<nThreadsCpu; t++)
		 pthread_join( thread1[t], NULL);

	bench.finish = gettimeInMilli();
	bench.finishF = (bench.finish - bench.start)/1000.; 

	// TODO: The total number of neurons is wrong
	tInfo->sharedData->neuronInfoWriter->writeResultsToFile(mode, nNeuronsTotal, tInfo->nComp[0], simType, bench);
	//delete tInfo->sharedData->neuronInfoWriter;

	delete[] tInfo->nNeurons;
	delete[] tInfo->nComp;
	delete tInfo;
	delete[] tInfoArray;

	printf ("Finished Simulation!!!\n");

	return 0;
}

// Used only to check the number of spikes joined in the HashMap
int spkTotal = 0;
int spkEqual = 0;

