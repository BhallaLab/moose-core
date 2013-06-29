#include "HinesMatrix.hpp"
#include "PlatformFunctions.hpp"
#include "HinesStruct.hpp"
#include "Connections.hpp"
#include "SpikeStatistics.hpp"
#include "GpuSimulationControl.hpp"

#include "SynapticData.hpp"
#include "KernelInfo.hpp"
#include "ThreadInfo.hpp"
#include "SharedNeuronGpuData.hpp"


#include <cassert>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <pthread.h>

#include <cuda.h> // Necessary to allow better eclipse integration
#include <cuda_runtime_api.h> // Necessary to allow better eclipse integration
#include <device_launch_parameters.h> // Necessary to allow better eclipse integration
#include <device_functions.h> // Necessary to allow better eclipse integration

extern __global__ void solveMatrixG(HinesStruct *hList, int nSteps, int nNeurons, ftype *vmListGlobal);

void checkCUDAError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err)
    {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) );

	exit(-1);
    }
}

//===================================================================================================

GpuSimulationControl::GpuSimulationControl(ThreadInfo *tInfo) {

	this->tInfo = tInfo;
	this->sharedData = tInfo->sharedData;
	this->kernelInfo = tInfo->sharedData->kernelInfo;
}

void GpuSimulationControl::prepareSynapses() {

	ftype spikeMem = 0;

	int *nNeurons = tInfo->nNeurons;
	SynapticData* synData = sharedData->synData;

	/**
	 * Prepare the synaptic channels and spike generation
	 */
	int totalTypes = synData->totalTypes;

	pthread_mutex_lock (sharedData->mutex);
	if (synData->activationListGlobal == 0) {

		synData->activationListGlobal    = (ftype **) malloc (sizeof(ftype *) * totalTypes); //*
		synData->activationListPosGlobal = (ucomp **) malloc (sizeof(ucomp *) * totalTypes); //*
		synData->activationListDevice    = (ftype **) malloc (sizeof(ftype *) * totalTypes); //*
		synData->activationListPosDevice = (ucomp **) malloc (sizeof(ucomp *) * totalTypes); //*

		synData->vmListHost    = (ftype **) malloc (sizeof(ftype *) * totalTypes);
		synData->vmListDevice  = (ftype **) malloc (sizeof(ftype *) * totalTypes);

		/*
		 * Used in the CPU and GPU version to distribute the spike list among the processes
		 */
		synData->genSpikeTimeListHost     = (ftype **) malloc (sizeof(ftype *)  * totalTypes);
		synData->nGeneratedSpikesHost     = (ucomp **) malloc (sizeof(ucomp *)  * totalTypes);

		synData->genSpikeTimeListDevice   = (ftype **) malloc (sizeof(ftype *)  * totalTypes);
		synData->nGeneratedSpikesDevice   = (ucomp **) malloc (sizeof(ucomp *)  * totalTypes);

		synData->genSpikeTimeListGpusDev  = (ftype ***) malloc (sizeof(ftype **) * sharedData->nThreadsCpu);
		synData->genSpikeTimeListGpusHost = (ftype ***) malloc (sizeof(ftype **) * sharedData->nThreadsCpu);
		synData->nGeneratedSpikesGpusDev  = (ucomp ***) malloc (sizeof(ucomp **) * sharedData->nThreadsCpu);
		synData->nGeneratedSpikesGpusHost = (ucomp ***) malloc (sizeof(ucomp **) * sharedData->nThreadsCpu);
	}
	pthread_mutex_unlock (sharedData->mutex);


	/**
	 * Prepare the delivered spike related lists
	 * - spikeListPos and spikeListSize
	 */
	for (int type = tInfo->startTypeThread; type < tInfo->endTypeThread; type++) {

		SynapticChannels *syn0 = sharedData->matrixList[type][0].synapticChannels;
		int globalActListSize = syn0->activationListSize * syn0->synapseListSize * nNeurons[type];
		synData->activationListGlobal[type] = (ftype *) malloc (sizeof(ftype) * globalActListSize); //*
		cudaMalloc ((void **) &(synData->activationListDevice[type]), sizeof(ftype) * globalActListSize); //*
		for (int i=0; i < globalActListSize; i++)
			synData->activationListGlobal[type][i] = 0;
		cudaMemcpy (synData->activationListDevice[type], synData->activationListGlobal[type],
				sizeof(ftype) * globalActListSize, cudaMemcpyHostToDevice); //*


		synData->activationListPosGlobal[type] = (ucomp *) malloc (sizeof(ucomp) * syn0->synapseListSize * nNeurons[type]); //*
		cudaMalloc ((void **) &(synData->activationListPosDevice[type]), sizeof(ucomp) * syn0->synapseListSize * nNeurons[type]); //*
		for (int i=0; i<syn0->synapseListSize * nNeurons[type]; i++)
			synData->activationListPosGlobal[type][i] = 0;
		cudaMemcpy (synData->activationListPosDevice[type], synData->activationListPosGlobal[type],
				sizeof(ucomp) * syn0->synapseListSize * nNeurons[type], cudaMemcpyHostToDevice); //*


		synData->vmListHost[type] = (ftype *) malloc(sizeof(ftype) * nNeurons[type]);
		cudaMalloc ((void **) &(synData->vmListDevice[type]), sizeof(ftype)  * nNeurons[type]);

		spikeMem += sizeof(ftype) * (globalActListSize + nNeurons[type]) + sizeof(ucomp) * syn0->synapseListSize * nNeurons[type];
	}

	/**
	 * Prepare the lists containing the generated spikes during each kernel call
	 */
	for (int type = tInfo->startTypeThread; type < tInfo->endTypeThread; type++) {
		int spikeTimeListSize = GENSPIKETIMELIST_SIZE;

		synData->genSpikeTimeListHost[type] = (ftype *) malloc(sizeof(ftype) * spikeTimeListSize * nNeurons[type]);
		cudaMalloc ((void **) &(synData->genSpikeTimeListDevice[type]), sizeof(ftype) * spikeTimeListSize * nNeurons[type]);

		synData->nGeneratedSpikesHost[type] = (ucomp *) malloc(sizeof(ucomp) * nNeurons[type]);
		cudaMalloc ((void **) &(synData->nGeneratedSpikesDevice[type]), sizeof(ucomp) * nNeurons[type]);

		int synapseListSize = sharedData->matrixList[type][0].synapticChannels->synapseListSize;

		for (int neuron = 0; neuron < nNeurons[type]; neuron++ ) {
			HinesStruct & h = sharedData->hList[type][neuron];
			h.spikeTimes  = synData->genSpikeTimeListDevice[type] + spikeTimeListSize * neuron;
			h.nGeneratedSpikes = synData->nGeneratedSpikesDevice[type];// + neuron;

			h.activationList = synData->activationListDevice[type]; // global list
			h.activationListPos = synData->activationListPosDevice[type] + synapseListSize * neuron;
		}

		spikeMem += sizeof(ftype) * spikeTimeListSize * nNeurons[type] + sizeof(ucomp) * nNeurons[type];
	}

	printf("Memory for Synapses: %10.3f MB.\n", spikeMem/(1024.*1024.));

}

void GpuSimulationControl::copyActivationListToGpu(int type) {

	int globalActListSize = sharedData->hList[type][0].synapseListSize * sharedData->hList[type][0].activationListSize * tInfo->nNeurons[type];
	cudaMemcpy(sharedData->hList[type][0].activationList, sharedData->synData->activationListGlobal[type],
			sizeof(ftype) * globalActListSize, cudaMemcpyHostToDevice);

}

void GpuSimulationControl::copyActivationListFromGpu() {

	SynapticData *synData = sharedData->synData;

	for (int type = tInfo->startTypeThread; type < tInfo->endTypeThread; type++) {

		int globalActListSize = sharedData->hList[type][0].synapseListSize * sharedData->hList[type][0].activationListSize * tInfo->nNeurons[type];
		cudaMemcpy(synData->activationListGlobal[type], sharedData->hList[type][0].activationList,
				sizeof(ftype) * globalActListSize, cudaMemcpyDeviceToHost);

		SynapticChannels *synChannel = sharedData->matrixList[type][0].synapticChannels;
		cudaMemcpy(synData->activationListPosGlobal[type], sharedData->hList[type][0].activationListPos,
				sizeof(ucomp) * synChannel->synapseListSize * tInfo->nNeurons[type], cudaMemcpyDeviceToHost);

	}

}

void GpuSimulationControl::transferHinesStructToGpu() {

	for(int type = tInfo->startTypeThread;type < tInfo->endTypeThread;type++){

		cudaMalloc((void**)((((&(sharedData->hGpu[type]))))), sizeof (HinesStruct) * tInfo->nNeurons[type]);
		cudaMemcpy(sharedData->hGpu[type], sharedData->hList[type], sizeof (HinesStruct) * tInfo->nNeurons[type], cudaMemcpyHostToDevice);
		checkCUDAError("Memory Allocation [hGPU]:");
	}
}


int GpuSimulationControl::prepareExecution(int type) {

	int nNeurons = tInfo->nNeurons[type];
	int nKernelSteps = kernelInfo->nKernelSteps;

	HinesStruct **hListPtr = &(sharedData->hList[type]);

	HinesStruct *hList = (HinesStruct *)malloc(nNeurons*sizeof(HinesStruct)); //new HinesStruct[nNeurons];

	HinesMatrix & m0 = sharedData->matrixList[type][0];
	int nComp = m0.nComp;
	int nCompActive = m0.activeChannels->getCompListSize();
	int nSynaptic = m0.synapticChannels->synapseListSize;

	/******************************************************************************************
	 * Allocates the ftype memory for all neurons and copies data to device
	 *******************************************************************************************/
	int fSharedMemMatrixSize    = sizeof(ftype) * (3*nComp + m0.mulListSize + m0.leftListSize); //+ nComp*nComp;
	int fSharedMemSynapticSize  = sizeof(ftype) * nSynaptic * SYN_CONST_N;
	int fSharedMemSize = fSharedMemMatrixSize + fSharedMemSynapticSize;

	int fExclusiveMemMatrixSize   = sizeof(ftype) * (5*nComp + m0.leftListSize);
	int fExclusiveMemActiveSize   = sizeof(ftype) * m0.activeChannels->ftypeMemSize;
	int fExclusiveMemSynapticSize = sizeof(ftype) * nSynaptic * (SYN_STATE_N + m0.synapticChannels->activationListSize);
	int fExclusiveMemSize = fExclusiveMemMatrixSize + fExclusiveMemActiveSize + fExclusiveMemSynapticSize + sizeof(ftype)*nComp*nKernelSteps;

	ftype *fMemory;
	cudaMalloc((void **)(&(fMemory)), fSharedMemSize + fExclusiveMemSize * nNeurons);
	ftype *fSharedMemMatrixAddress = m0.Cm;
	ftype *fSharedMemSynapticAddress = m0.synapticChannels->synConstants;

	cudaMemcpy(fMemory, fSharedMemMatrixAddress, fSharedMemMatrixSize, cudaMemcpyHostToDevice);
	cudaMemcpy(fMemory+fSharedMemMatrixSize/sizeof(ftype), fSharedMemSynapticAddress, fSharedMemSynapticSize, cudaMemcpyHostToDevice);


	/******************************************************************************************
	 * Allocates the ucomp memory for all neurons and copies data to device
	 *******************************************************************************************/

	int uExclusiveMemSynapticSize = sizeof(ucomp) * nSynaptic;
	ucomp *uExclusiveMemSynaptic;
	cudaMalloc((void **)(&(uExclusiveMemSynaptic)), uExclusiveMemSynapticSize * nNeurons);

	int uMemMatrixSize    = sizeof(ucomp) * ((m0.mulListSize + m0.leftListSize) * 2 + nComp);
	int uMemActiveSize    = sizeof(ucomp) * m0.activeChannels->ucompMemSize;
	int uMemSynapticSize  = sizeof(ucomp) * 2 * nSynaptic;
	int uMemSize = uMemMatrixSize + uMemActiveSize + uMemSynapticSize;

	ucomp *uMemory;
	cudaMalloc((void **)(&(uMemory)), uMemSize);
	ucomp *uMemActiveAddress   = uMemory + uMemMatrixSize / sizeof(ucomp);
	ucomp *uMemSynapticAddress = uMemActiveAddress + uMemActiveSize / sizeof(ucomp);

	cudaMemcpy(uMemory,             m0.ucompMemory,                       uMemMatrixSize,   cudaMemcpyHostToDevice);
	cudaMemcpy(uMemActiveAddress,   m0.activeChannels->ucompMem,          uMemActiveSize,   cudaMemcpyHostToDevice);
	cudaMemcpy(uMemSynapticAddress, m0.synapticChannels->synapseCompList, uMemSynapticSize, cudaMemcpyHostToDevice);

	//cudaMemcpy(uMemSynapticAddress, m0.synapticChannels->synapseCompList, uMemSynapticSize, cudaMemcpyHostToDevice);
	//cudaMemcpy(uMemActiveAddress,   m0.activeChannels->getCompList(),     uMemActiveSize,   cudaMemcpyHostToDevice); // TODO: old active

	printf("Memory for Neurons: %10.3f MB for %d neurons of type %d.\n",(fSharedMemSize + uMemSize + (fExclusiveMemSize +  uExclusiveMemSynapticSize) * nNeurons)/(1024.*1024.), nNeurons, type);

	/******************************************************************************************
	 * Prepare the MatrixStruct h for each neuron in the GPU
	 *******************************************************************************************/
	for (int neuron = 0; neuron < nNeurons; neuron++ ) {

		HinesMatrix & m = sharedData->matrixList[type][neuron];
		HinesStruct & h = hList[neuron];

		/****************************************************
		 * Fields of the HinesStruct
		 ****************************************************/
		h.currStep = m.currStep;
		h.vRest = m.vRest;
		h.dx = m.dx;
		h.nComp = m.nComp;
		h.dt = m.dt;
		h.triangAll = m.triangAll;
		h.mulListSize = m.mulListSize;
		h.leftListSize = m.leftListSize;
		h.type = type;
		h.nNeurons = nNeurons;

		/****************************************************
		 * ftype memory shared among all neurons
		 ****************************************************/
		h.memoryS = fMemory;
		h.Cm = h.memoryS;
		h.Ra = h.Cm + nComp;
		h.Rm = h.Ra + nComp;
		h.leftList = h.Rm + nComp;
		h.mulList  = h.leftList + m.leftListSize; // Used only when triangAll = 0

		/****************************************************
		 * ftype memory allocated per neuron
		 ****************************************************/
		h.memoryE = fMemory + fSharedMemSize/sizeof(ftype) + neuron*fExclusiveMemSize/sizeof(ftype);
		ftype *exclusiveAddressM = m.rhsM;
		cudaMemcpy(h.memoryE, exclusiveAddressM, fExclusiveMemMatrixSize, cudaMemcpyHostToDevice);
		// must match the order in HinesMatrix.cpp
		h.rhsM = h.memoryE	;
		h.vmList = h.rhsM + nComp;
		h.vmTmp = h.vmList + nComp;
		h.curr = h.vmTmp + nComp;
		h.active = h.curr + nComp;
		h.triangList = h.active + nComp; // triangularized list
		h.vmTimeSerie = h.triangList + m.leftListSize;

		/****************************************************
		 * ucomp memory shared among all neurons
		 ****************************************************/
		h.mulListComp    = uMemory;
		h.mulListDest    = h.mulListComp  + h.mulListSize;
		h.leftListLine   = h.mulListDest  + h.mulListSize;
		h.leftListColumn = h.leftListLine + h.leftListSize;
		h.leftStartPos   = h.leftListColumn + h.leftListSize;


		/****************************************************
		 * Active channels using the old  and new implementations
		 ****************************************************/
		if (nCompActive > 0 && m.activeChannels->channelInfo == 0) {

		}
		else if (m.activeChannels->channelInfo != 0) {

			ftype *activeMemAddress = h.vmTimeSerie + nComp*nKernelSteps;
			cudaMemcpy(activeMemAddress, m.activeChannels->ftypeMem, fExclusiveMemActiveSize, cudaMemcpyHostToDevice);

			h.nChannels    = m.activeChannels->nChannels;
			h.compListSize = m.activeChannels->nActiveComp;
			h.nGatesTotal  = m.activeChannels->nGatesTotal;

			h.channelEk   = activeMemAddress;
			h.channelGbar = h.channelEk   + h.nChannels;
			h.eLeak       = h.channelGbar + h.nChannels;
			h.gActive	  = h.eLeak       + h.compListSize;
			h.gateState   = h.gActive	  + h.compListSize;
			h.gatePar     = h.gateState   + h.nGatesTotal;


			h.compList    = uMemActiveAddress;
			h.channelInfo = h.compList    + h.compListSize;
			h.gateInfo    = h.channelInfo + (h.nChannels * N_CHANNEL_FIELDS);
		}
		checkCUDAError("Memory Allocation after active:");

		/****************************************************
		 * Synaptic Channels
		 ****************************************************/
		if (m.synapticChannels != 0) {

			SynapticChannels *synChan = m.synapticChannels;
			h.synapseListSize    = synChan->synapseListSize;
			h.activationListSize = synChan->activationListSize;

			h.synapseCompList = uMemSynapticAddress;
			h.synapseTypeList = h.synapseCompList + h.synapseListSize;

			h.synConstants = fMemory + fSharedMemMatrixSize/sizeof(ftype);

			// exclusive fmemory
			h.synState = h.gatePar + h.nGatesTotal * N_GATE_FUNC_PAR;
			cudaMemcpy(h.synState, synChan->synState,
					sizeof(ftype) * h.synapseListSize * SYN_STATE_N, cudaMemcpyHostToDevice);

			h.activationList = h.synState + h.synapseListSize * SYN_STATE_N; // h.synapseListSize * h.activationListSize;
			// TODO: should not be necessary
			cudaMemcpy(h.activationList, synChan->activationList,
					sizeof(ftype) * h.synapseListSize * h.activationListSize, cudaMemcpyHostToDevice);

			h.activationListPos = uExclusiveMemSynaptic + neuron * uExclusiveMemSynapticSize/sizeof(ucomp);
			cudaMemcpy(h.activationListPos, synChan->activationListPos, uExclusiveMemSynapticSize, cudaMemcpyHostToDevice);

			// Used for spike generation
			h.lastSpike 		= m.lastSpike;
			h.spikeTimeListSize = m.spikeTimeListSize;
			h.threshold         = m.threshold;
			h.minSpikeInterval  = m.minSpikeInterval;
		}
		checkCUDAError("Memory Allocation after synaptic:");

		//if (benchConf.simCommMode == NN_GPU)
		sharedData->matrixList[type][neuron].freeMem();
	}

	*hListPtr = hList;

	return 0;
}

void GpuSimulationControl::performGpuNeuronalProcessing() {

	for (int type = tInfo->startTypeThread; type < tInfo->endTypeThread; type++) {

		int nThreadsProc =  tInfo->nNeurons[type]/kernelInfo->nBlocksProc[type];
		if (tInfo->nNeurons[type] % kernelInfo->nBlocksProc[type]) nThreadsProc++;

		//printf("launching kernel for type %d...\n", type);
		SynapticData *synData = sharedData->synData;


		solveMatrixG<<<kernelInfo->nBlocksProc[type], nThreadsProc, kernelInfo->sharedMemSizeProc>>>(
				sharedData->hGpu[type], kernelInfo->nKernelSteps, tInfo->nNeurons[type], synData->vmListDevice[type]);

		checkCUDAError("After SolveMatrixG Kernel:");
	}

}

void GpuSimulationControl::checkVmValues()
{
	SynapticData *synData = sharedData->synData;
    for (int type = tInfo->startTypeThread; type < tInfo->endTypeThread; type++)
					for (int n = 0; n < tInfo->nNeurons[type]; n++)
						if ( synData->vmListHost[type][n] < -500 || 500 < synData->vmListHost[type][n] || synData->vmListHost[type][n] == 0.000000000000) {
							printf("POSSIBLE ERROR: ********* type=%d neuron=%d Vm=%.2f\n", type, n, synData->vmListHost[type][n]);
							//assert(false);
						}
}



void GpuSimulationControl::prepareGpuSpikeDeliveryStructures()
{
	SynapticData *synData = sharedData->synData;
	int threadNumber = tInfo->threadNumber;

    synData->genSpikeTimeListGpusHost[threadNumber] = (ftype**)(malloc(sizeof (ftype*) * tInfo->totalTypes));
    for (int type = 0; type < tInfo->totalTypes; type++) {
			int genSpikeTimeListSize = GENSPIKETIMELIST_SIZE;
			// The device memory for the types of the current thread are already allocated
			if (tInfo->startTypeThread <= type && type < tInfo->endTypeThread)
				synData->genSpikeTimeListGpusHost[threadNumber][type] = synData->genSpikeTimeListDevice[type];
			else
				cudaMalloc ((void **) &(synData->genSpikeTimeListGpusHost[threadNumber][type]),
						sizeof(ftype) * genSpikeTimeListSize * tInfo->nNeurons[type]);
		}
    // Copies the list of pointers to the genSpikeLists of each type
    cudaMalloc((void**)(&synData->genSpikeTimeListGpusDev[threadNumber]), sizeof (ftype*) * tInfo->totalTypes);
    cudaMemcpy(synData->genSpikeTimeListGpusDev[threadNumber], synData->genSpikeTimeListGpusHost[threadNumber], sizeof (ftype*) * tInfo->totalTypes, cudaMemcpyHostToDevice);
    synData->nGeneratedSpikesGpusHost[threadNumber] = (ucomp**)(malloc(sizeof (ucomp*) * tInfo->totalTypes));
    for(int type = 0;type < tInfo->totalTypes;type++){
        if(tInfo->startTypeThread <= type && type < tInfo->endTypeThread)
            synData->nGeneratedSpikesGpusHost[threadNumber][type] = synData->nGeneratedSpikesDevice[type];

        else
            cudaMalloc((void**)(&(synData->nGeneratedSpikesGpusHost[threadNumber][type])), sizeof (ucomp) * tInfo->nNeurons[type]);

    }
    // Copies the list of pointers to the nGeneratedSpikes of each type
    cudaMalloc((void**)(&synData->nGeneratedSpikesGpusDev[threadNumber]), sizeof (ucomp*) * tInfo->totalTypes);
    cudaMemcpy(synData->nGeneratedSpikesGpusDev[threadNumber], synData->nGeneratedSpikesGpusHost[threadNumber], sizeof (ucomp*) * tInfo->totalTypes, cudaMemcpyHostToDevice);
}

void GpuSimulationControl::readGeneratedSpikesFromGPU()
{
    /*--------------------------------------------------------------
     * Reads information from spike sources
     *--------------------------------------------------------------*/
    if(benchConf.verbose == 1)
        printf("Getting spikes %d\n", tInfo->threadNumber);

    for(int type = tInfo->startTypeThread;type < tInfo->endTypeThread;type++){

        cudaMemcpy(sharedData->synData->genSpikeTimeListHost[type], sharedData->synData->genSpikeTimeListDevice[type],
        		sizeof (ftype) * sharedData->hList[type][0].spikeTimeListSize * tInfo->nNeurons[type], cudaMemcpyDeviceToHost);

        cudaMemcpy(sharedData->synData->nGeneratedSpikesHost[type], sharedData->synData->nGeneratedSpikesDevice[type],
        		sizeof (ucomp) * tInfo->nNeurons[type], cudaMemcpyDeviceToHost);

        checkCUDAError("Synapses2:");
    }

}

void GpuSimulationControl::configureGpuKernel()
{

    /*--------------------------------------------------------------
	 * Select the device attributed to each thread
	 *--------------------------------------------------------------*/
    int nDevices = 0;
    cudaGetDeviceCount(&nDevices);
	if (nDevices == 0) {
		printf("ERROR: No valid CUDA devices found. Exiting...\n");
		exit(-1);
	}


    cudaSetDevice((tInfo->threadNumber + 1) % nDevices);
    cudaGetDevice(&(tInfo->deviceNumber));
    tInfo->prop = new struct cudaDeviceProp;
    cudaGetDeviceProperties(tInfo->prop, tInfo->deviceNumber);

    checkCUDAError("Device selection:");
    //--------------------------------------------------------------
    /*--------------------------------------------------------------
	 * Configure number of threads and shared memory size for each kernel
	 *--------------------------------------------------------------*/
    kernelInfo->maxThreadsProc = 64;
    kernelInfo->maxThreadsComm = 16;
    kernelInfo->sharedMemSizeProc = 15 * 1024; // Valid for capability 1.x (16kB)
    kernelInfo->sharedMemSizeComm = 15 * 1024; // Valid for capability 1.x (16kB)
    if(tInfo->prop->major == 2){
        kernelInfo->maxThreadsProc = 256;
        kernelInfo->maxThreadsComm = 32; // or can be 64
        kernelInfo->sharedMemSizeProc = 47 * 1024; // Valid for capability 2.x (48kB)
        kernelInfo->sharedMemSizeComm = 15 * 1024; // Valid for capability 2.x (48kB)
    }
    else if(tInfo->prop->major == 3){
      kernelInfo->maxThreadsProc = 384;//384;
      kernelInfo->maxThreadsComm = 32; // or can be 64
      kernelInfo->sharedMemSizeProc = 47 * 1024; // Valid for capability 2.x (48kB)
      kernelInfo->sharedMemSizeComm = 15 * 1024; // Valid for capability 2.x (48kB)
    }
    //--------------------------------------------------------------
    for(int type = tInfo->startTypeThread;type < tInfo->endTypeThread;type++){
        // Number of blocks: multiple of #GPU multiprocessors and respects maxThreadsProc condition
        kernelInfo->nBlocksProc[type] = tInfo->prop->multiProcessorCount * (tInfo->nNeurons[type] / kernelInfo->maxThreadsProc / tInfo->prop->multiProcessorCount);
        if(tInfo->nNeurons[type] % kernelInfo->maxThreadsProc != 0 || (tInfo->nNeurons[type] / kernelInfo->maxThreadsProc) % kernelInfo->maxThreadsProc != 0)
            kernelInfo->nBlocksProc[type] += tInfo->prop->multiProcessorCount;

    }

    for(int destType = tInfo->startTypeThread;destType < tInfo->endTypeThread;destType++){

    	// Number of blocks: multiple of #GPU multiprocessors and respects maxThreadsComm condition
    	kernelInfo->nBlocksComm[destType] = tInfo->prop->multiProcessorCount * (tInfo->nNeurons[destType] / kernelInfo->maxThreadsComm / tInfo->prop->multiProcessorCount);
        if(tInfo->nNeurons[destType] % kernelInfo->maxThreadsComm != 0 || (tInfo->nNeurons[destType] / kernelInfo->maxThreadsComm) % kernelInfo->maxThreadsComm != 0)
        	kernelInfo->nBlocksComm[destType] += tInfo->prop->multiProcessorCount;
    }

    //	if (nComp0 <= 4) nThreads = (sizeof (ftype) == 4) ? 196 : 96;
    //	else if (nComp0 <= 8) nThreads = (sizeof (ftype) == 4) ? 128 : 64;
    //	else if (nComp0 <= 12) nThreads = (sizeof (ftype) == 4) ? 96 : 32;
    //	else if (nComp0 <= 16) nThreads = (sizeof (ftype) == 4) ? 64 : 32;

}

void GpuSimulationControl::updateSharedDataInfo()
{
    sharedData->hList = new HinesStruct*[tInfo->totalTypes];
    sharedData->hGpu = new HinesStruct*[tInfo->totalTypes];
    sharedData->synData = new SynapticData;
    sharedData->synData->activationListGlobal = 0;
    sharedData->synData->totalTypes = tInfo->totalTypes;
    sharedData->neuronInfoWriter = new NeuronInfoWriter(tInfo);
    sharedData->spkStat = new SpikeStatistics(tInfo->nNeurons, tInfo->nTypes, tInfo->totalTypes, tInfo->sharedData->typeList);
    kernelInfo->nThreadsComm = new int[tInfo->totalTypes];
    kernelInfo->nBlocksComm = new int[tInfo->totalTypes];
    kernelInfo->nBlocksProc = new int[tInfo->totalTypes];
}



void GpuSimulationControl::addToInterleavedSynapticActivationList(
		ftype *activationList, ucomp *activationListPos, int activationListSize, int neuron, int nNeurons,
		ftype currTime, ftype dt, ucomp synapse, ftype spikeTime, ftype delay, ftype weight) {

	ftype fpos = (spikeTime + delay - currTime) / dt;

	ucomp pos  = ( activationListPos[synapse] + (ucomp)fpos + 1 ) % activationListSize;
	pos       += synapse * activationListSize;

	ucomp nextPos  = ( pos + 1 ) % activationListSize;
	nextPos       += synapse * activationListSize;

	ftype diff = fpos - (int)fpos;

	// TODO: race conditions can occur here with multiple threads
	activationList[    pos * nNeurons + neuron] += (weight / dt) * ( 1 - diff );
	activationList[nextPos * nNeurons + neuron] += (weight / dt) * diff;
}
