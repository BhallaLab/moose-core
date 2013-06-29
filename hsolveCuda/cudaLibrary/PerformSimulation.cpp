#include <cstdio>
#include <cassert>
#include <cstdlib>

#include "PerformSimulation.hpp"
#include "GpuSimulationControl.hpp"

#include "SharedNeuronGpuData.hpp"
#include "ThreadInfo.hpp"
#include "KernelInfo.hpp"
#include "SynapticData.hpp"

#include "Connections.hpp"
#include "HinesMatrix.hpp"
#include "ActiveChannels.hpp"
#include "PlatformFunctions.hpp"

//#include "HinesStruct.hpp"
#include "SpikeStatistics.hpp"

#include <cmath>
#include <unistd.h>

#include <cuda_runtime_api.h> // Necessary to allow better eclipse integration

PerformSimulation::PerformSimulation(struct ThreadInfo *tInfo) {

	this->tInfo      = tInfo;
	this->sharedData = tInfo->sharedData;
	this->kernelInfo = tInfo->sharedData->kernelInfo;
}

void PerformSimulation::createActivationLists( ) {

	int listSize = sharedData->maxDelay / sharedData->dt;

	for (int type = tInfo->startTypeThread; type < tInfo->endTypeThread; type++)
		for (int target = 0; target < tInfo->nNeurons[type]; target++)
			sharedData->matrixList[type][target].synapticChannels->configureSynapticActivationList( sharedData->dt, listSize );
}

void PerformSimulation::createNeurons( ftype dt ) {


	SharedNeuronGpuData *sharedData = tInfo->sharedData;


    /**------------------------------------------------------------------------------------
	 * Creates the neurons that will be simulated by the threads
	 *-------------------------------------------------------------------------------------*/
    for(int type = tInfo->startTypeThread;type < tInfo->endTypeThread;type++){
        int nComp = tInfo->nComp[type];
        int nNeurons = tInfo->nNeurons[type];

        sharedData->matrixList[type] = new HinesMatrix[nNeurons];

        for(int n = 0;n < nNeurons;n++){
            HinesMatrix & m = sharedData->matrixList[type][n];
            if(nComp == 1)
                m.defineNeuronCableSquid();

            else
                m.defineNeuronTreeN(nComp, 1);

            m.createTestMatrix();
            m.dt     = dt;
            m.neuron = n;
            m.type   = type;
        }
    }

}

void PerformSimulation::initializeThreadInformation(){

	SharedNeuronGpuData *sharedData = tInfo->sharedData;

	pthread_mutex_lock (sharedData->mutex);
	tInfo->threadNumber = sharedData->nBarrier;
	sharedData->nBarrier++;
	if (sharedData->nBarrier < sharedData->nThreadsCpu)
		pthread_cond_wait(sharedData->cond, sharedData->mutex);
	else {
		sharedData->nBarrier = 0;
		pthread_cond_broadcast(sharedData->cond);
	}
	pthread_mutex_unlock (sharedData->mutex);

	char *randstate = new char[256];
	sharedData->randBuf[tInfo->threadNumber] = (struct random_data*)calloc(1, sizeof(struct random_data));
	initstate_r(tInfo->sharedData->globalSeed + tInfo->threadNumber,
			randstate, 256, tInfo->sharedData->randBuf[tInfo->threadNumber]);

	int nThreadsCpu 	= tInfo->sharedData->nThreadsCpu;
	int nTypesPerThread = (tInfo->totalTypes / nThreadsCpu);
	tInfo->startTypeThread = (tInfo->threadNumber ) * nTypesPerThread;
	tInfo->endTypeThread = (tInfo->threadNumber + 1 ) * nTypesPerThread;

	int typeProcessCurr = 0;
	tInfo->typeProcess = new int[tInfo->totalTypes];
	for(int type = 0;type < tInfo->totalTypes;type++){
		if(type / ((typeProcessCurr + 1) * nThreadsCpu * nTypesPerThread) == 1)
			typeProcessCurr++;

		tInfo->typeProcess[type] = typeProcessCurr;
	}


}

void PerformSimulation::updateBenchmark()
{
	bench.totalHinesKernel	+= (bench.kernelFinish 	- bench.kernelStart)/1000.;
	bench.totalConnRead	  	+= (bench.connRead 		- bench.kernelFinish)/1000.;
	bench.totalConnWait		+= (bench.connWait 		- bench.connRead)/1000.;
	bench.totalConnWrite	+= (bench.connWrite 	- bench.connWait)/1000.;
}

void PerformSimulation::syncCpuThreads()
{
    pthread_mutex_lock(sharedData->mutex);
    sharedData->nBarrier++;
    if(sharedData->nBarrier < sharedData->nThreadsCpu)
        pthread_cond_wait(sharedData->cond, sharedData->mutex);

    else{
        sharedData->nBarrier = 0;
        pthread_cond_broadcast(sharedData->cond);
    }
    pthread_mutex_unlock(sharedData->mutex);
}

void PerformSimulation::updateGenSpkStatistics(int *& nNeurons, SynapticData *& synData)
{
	/*--------------------------------------------------------------
	 * Used to print spike statistics in the end of the simulation
	 *--------------------------------------------------------------*/
	for (int type=tInfo->startTypeThread; type < tInfo->endTypeThread; type++)
		for (int c=0; c<nNeurons[type]; c++)
			sharedData->spkStat->addGeneratedSpikes(type, c, NULL, synData->nGeneratedSpikesHost[type][c]);
}

void PerformSimulation::generateRandomSpikes( int type, RandomSpikeInfo & randomSpkInfo )
{

	ftype currTime    = sharedData->dt * (tInfo->kStep + kernelInfo->nKernelSteps);
	ftype randWeight  = sharedData->randWeight;
	ucomp randSynapse = 0;

	randomSpkInfo.listSize =
			3 * sharedData->inputSpikeRate * kernelInfo->nKernelSteps *
			sharedData->dt * tInfo->nNeurons[type];
	if (randomSpkInfo.listSize < 200)
	  randomSpkInfo.listSize = 200;
	//printf ("randomSpkInfo.listSize=%d\n",randomSpkInfo.listSize);

	randomSpkInfo.spikeTimes = new ftype[ randomSpkInfo.listSize ];
	randomSpkInfo.spikeDest = new int[ randomSpkInfo.listSize ];

	int kernelSteps = kernelInfo->nKernelSteps;
	ftype dt = sharedData->dt;

	randomSpkInfo.nRandom = 0;
	for (int neuron = 0; neuron < tInfo->nNeurons[type]; neuron++) {
		HinesMatrix & m = sharedData->matrixList[type][neuron];

		if ((tInfo->kStep + kernelSteps)*m.dt > 9.9999 ){ //&& sharedData->typeList[type] == PYRAMIDAL_CELL) {
			int32_t randValue;
			random_r(sharedData->randBuf[tInfo->threadNumber], &randValue);
			ftype rate = (sharedData->inputSpikeRate) * (kernelSteps * dt);
			ftype kPos = (ftype)randValue/RAND_MAX;
			if ( kPos < rate ) {
				ftype spkTime = currTime + (int)( kPos * kernelSteps ) * dt;

				randomSpkInfo.nRandom++;

				GpuSimulationControl::addToInterleavedSynapticActivationList(
						sharedData->synData->activationListGlobal[type],
						sharedData->synData->activationListPosGlobal[type] + neuron * m.synapticChannels->synapseListSize,
						m.synapticChannels->activationListSize,
						neuron, tInfo->nNeurons[type], currTime, sharedData->dt, randSynapse, spkTime, 0, randWeight);



			}
		}
	}
}

void PerformSimulation::addReceivedSpikesToTargetChannelCPU()
{

	ftype currTime = sharedData->dt * (tInfo->kStep + kernelInfo->nKernelSteps);

	ConnectionInfo *connInfo = sharedData->connInfo;

	int conn 	= tInfo->threadNumber       * connInfo->nConnections/sharedData->nThreadsCpu;
	int endConn = (tInfo->threadNumber + 1) * connInfo->nConnections/sharedData->nThreadsCpu;
	if (tInfo->threadNumber == sharedData->nThreadsCpu-1)
		endConn = connInfo->nConnections;

	for ( ; conn < endConn; conn++) {

		int dType   = connInfo->dest[conn] / CONN_NEURON_TYPE;
		int dNeuron = connInfo->dest[conn]   % CONN_NEURON_TYPE;
		int sType   = connInfo->source[conn] / CONN_NEURON_TYPE;
		int sNeuron = connInfo->source[conn] % CONN_NEURON_TYPE;

		ucomp nGeneratedSpikes = sharedData->synData->nGeneratedSpikesHost[sType][sNeuron];
		if (nGeneratedSpikes > 0) {
			ftype *spikeTimes = sharedData->synData->genSpikeTimeListHost[sType] + GENSPIKETIMELIST_SIZE * sNeuron;

			SynapticChannels *targetSynapse = sharedData->matrixList[ dType ][ dNeuron ].synapticChannels;


			for (int spk=0; spk < nGeneratedSpikes; spk++) {

				GpuSimulationControl::addToInterleavedSynapticActivationList(
						sharedData->synData->activationListGlobal[dType],
						sharedData->synData->activationListPosGlobal[dType] + dNeuron * targetSynapse->synapseListSize,
						targetSynapse->activationListSize,
						dNeuron, tInfo->nNeurons[dType], currTime, sharedData->dt,
						connInfo->synapse[conn], spikeTimes[spk], connInfo->delay[conn], connInfo->weigth[conn]);
			}
		}
	}



	for (int type = tInfo->startTypeThread; type < tInfo->endTypeThread; type++) {
		for (int source = 0; source < tInfo->nNeurons[type]; source++) {
			// Used to print spike statistics in the end of the simulation
			sharedData->spkStat->addReceivedSpikes(type, source,
					sharedData->matrixList[type][source].synapticChannels->getAndResetNumberOfAddedSpikes());
		}
	}
}


/*======================================================================================================
 * Performs the execution
 *======================================================================================================*/
int PerformSimulation::launchExecution() {

	GpuSimulationControl *gpuSimulation = new GpuSimulationControl(tInfo);

	/**
	 * Initializes thread information
	 */
	initializeThreadInformation( );

	/**------------------------------------------------------------------------------------
	 * Creates the neurons that will be simulated by the threads
	 *-------------------------------------------------------------------------------------*/
    sharedData->dt = 0.1; // 0.1ms
    sharedData->minDelay = 10; // 10ms
    sharedData->maxDelay = 20; // 10ms
	kernelInfo->nKernelSteps = sharedData->minDelay / sharedData->dt;

    createNeurons(sharedData->dt);




    char hostname[50];
    gethostname(hostname, 50);

	printf("threadNumber = %d | types [%d|%d] | seed=%d | hostname=%s\n" ,
			tInfo->threadNumber, tInfo->startTypeThread, tInfo->endTypeThread-1, tInfo->sharedData->globalSeed, hostname);

    int *nNeurons = tInfo->nNeurons;
    int startTypeThread = tInfo->startTypeThread;
    int endTypeThread = tInfo->endTypeThread;
    int threadNumber = tInfo->threadNumber;

    if(threadNumber == 0)
    	gpuSimulation->updateSharedDataInfo();

    /*--------------------------------------------------------------
	 * Creates the connections between the neurons
	 *--------------------------------------------------------------*/
    if (threadNumber == 0) {
		sharedData->connection = new Connections();
		if (sharedData->connectivityType == CONNECT_RANDOM_1)
			sharedData->connection->connectRandom ( tInfo );
		else if (sharedData->connectivityType == CONNECT_RANDOM_2)
			sharedData->connection->connectRandom2 ( tInfo );
		else {
			printf("ERROR: Invalid connectivity type");
			exit(-1);
		}
		sharedData->connInfo = sharedData->connection->getConnectionInfo();
	}

    //Synchronize threads before starting
    syncCpuThreads();

    bench.matrixSetup = gettimeInMilli();
    bench.matrixSetupF = (bench.matrixSetup - bench.start) / 1000.;

    /*--------------------------------------------------------------
	 * Configure the Device and GPU kernel information
	 *--------------------------------------------------------------*/
    gpuSimulation->configureGpuKernel();

    /*--------------------------------------------------------------
	 * Initializes the benchmark counters
	 *--------------------------------------------------------------*/
    if(threadNumber == 0){
        bench.totalHinesKernel = 0;
        bench.totalConnRead = 0;
        bench.totalConnWait = 0;
        bench.totalConnWrite = 0;
    }

	createActivationLists();

    /*--------------------------------------------------------------
	 * Allocates the memory on the GPU for neuron information and transfers the data
	 *--------------------------------------------------------------*/
	for(int type = startTypeThread;type < endTypeThread;type++)
		gpuSimulation->prepareExecution(type);

    /*--------------------------------------------------------------
	 * Allocates the memory on the GPU for the communications and transfers the data
	 *--------------------------------------------------------------*/
	gpuSimulation->prepareSynapses();

    SynapticData *synData = sharedData->synData;
    int nKernelSteps = kernelInfo->nKernelSteps;

    /*--------------------------------------------------------------
	 * Sends the complete data to the GPUs
	 *--------------------------------------------------------------*/
   	gpuSimulation->transferHinesStructToGpu();

    /*--------------------------------------------------------------
	 * Guarantees that all connections have been setup
	 *--------------------------------------------------------------*/
    syncCpuThreads();


    /*--------------------------------------------------------------
	 * Prepare the lists of generated spikes used for GPU spike delivery
	 *--------------------------------------------------------------*/
   	gpuSimulation->prepareGpuSpikeDeliveryStructures();

    /*--------------------------------------------------------------
	 * Synchronize threads before beginning [Used only for Benchmarking]
	 *--------------------------------------------------------------*/
    syncCpuThreads();

    printf("Launching GPU kernel with %d blocks and %d (+1) threads per block for types %d-%d for thread %d "
    		"on device %d [%s|%d.%d|MP=%d|G=%dMB|S=%dkB].\n", kernelInfo->nBlocksProc[startTypeThread],
    		nNeurons[startTypeThread] / kernelInfo->nBlocksProc[startTypeThread], startTypeThread, endTypeThread - 1,
    		threadNumber, tInfo->deviceNumber, tInfo->prop->name, tInfo->prop->major, tInfo->prop->minor,
    		tInfo->prop->multiProcessorCount, (int)((tInfo->prop->totalGlobalMem / 1024 / 1024)),
    		(int)((tInfo->prop->sharedMemPerBlock / 1024)));

    if(threadNumber == 0){
        bench.execPrepare = gettimeInMilli();
        bench.execPrepareF = (bench.execPrepare - bench.matrixSetup) / 1000.;
    }

    /*--------------------------------------------------------------
	 * Solves the matrix for n steps
	 *--------------------------------------------------------------*/
    ftype dt = sharedData->dt;
    int nSteps = sharedData->totalTime / dt;

    for (tInfo->kStep = 0; tInfo->kStep < nSteps; tInfo->kStep += nKernelSteps) {

		// Synchronizes the thread to wait for the communication

		if (threadNumber == 0 && tInfo->kStep % 1000 == 0)
			printf("Starting Kernel %d -----------> %d \n", threadNumber, tInfo->kStep);

		if (threadNumber == 0) // Benchmarking
			bench.kernelStart  = gettimeInMilli();

		addReceivedSpikesToTargetChannelCPU();
		gpuSimulation->performGpuNeuronalProcessing();


		cudaThreadSynchronize();

		if (threadNumber == 0) // Benchmarking
			bench.kernelFinish = gettimeInMilli();

		/*--------------------------------------------------------------
		 * Reads information from spike sources fromGPU
		 *--------------------------------------------------------------*/
		gpuSimulation->readGeneratedSpikesFromGPU();

		/*--------------------------------------------------------------
		 * Synchronize threads before communication
		 *--------------------------------------------------------------*/
		syncCpuThreads();

		if (threadNumber == 0) {
			bench.connRead = gettimeInMilli();
			bench.connWait = gettimeInMilli();
		}

		/*--------------------------------------------------------------
		 * Adds the generated spikes to the target synaptic channel
		 * Used only for communication processing in the CPU
		 *--------------------------------------------------------------*/
		gpuSimulation->copyActivationListFromGpu();

		syncCpuThreads();

		// Used to print spike statistics in the end of the simulation
		updateGenSpkStatistics(nNeurons, synData);

		/*--------------------------------------------------------------
		 * Copy the Vm from GPUs to the CPU memory
		 *--------------------------------------------------------------*/
		if (benchConf.assertResultsAll == 1 || benchConf.printAllVmKernelFinish == 1)
			for (int type = startTypeThread; type < endTypeThread; type++)
				cudaMemcpy(synData->vmListHost[type], synData->vmListDevice[type], sizeof(ftype) * nNeurons[type], cudaMemcpyDeviceToHost);


		/*--------------------------------------------------------------
		 * Writes Vm to file at the end of each kernel execution
		 *--------------------------------------------------------------*/
		if (benchConf.assertResultsAll == 1)
			gpuSimulation->checkVmValues();

		/*--------------------------------------------------------------
		 * Check if Vm is ok for all neurons
		 *--------------------------------------------------------------*/
		if (threadNumber == 0 && benchConf.printAllVmKernelFinish == 1)
			sharedData->neuronInfoWriter->writeVmToFile(tInfo->kStep);

		/*-------------------------------------------------------
		 * Perform Communications
		 *-------------------------------------------------------*/
		for (int type = startTypeThread; type < endTypeThread; type++) {

			/*-------------------------------------------------------
			 *  Generates random spikes for the network
			 *-------------------------------------------------------*/
			struct RandomSpikeInfo randomSpkInfo;
			generateRandomSpikes(type, randomSpkInfo);

			/*-------------------------------------------------------
			 * Perform CPU and GPU Communications
			 *-------------------------------------------------------*/
			gpuSimulation->copyActivationListToGpu(type);

			delete []randomSpkInfo.spikeTimes;
			delete []randomSpkInfo.spikeDest;
		}

		if (threadNumber == 0)
			if (benchConf.gpuCommBenchMode == GPU_COMM_SIMPLE || benchConf.checkCommMode(NN_CPU) )
				bench.connWrite = gettimeInMilli();

		if (threadNumber == 0 && benchConf.printSampleVms == 1)
			sharedData->neuronInfoWriter->writeSampleVm(tInfo->kStep);

		if (benchConf.printAllSpikeTimes == 1)
			if (threadNumber == 0) // Uses only data from SpikeStatistics::addGeneratedSpikes
				sharedData->spkStat->printKernelSpikeStatistics((tInfo->kStep+nKernelSteps)*dt);

		if (threadNumber == 0)
			updateBenchmark();


    }
    // --------------- Finished the simulation ------------------------------------

    if (threadNumber == 0) {
    	bench.execExecution  = gettimeInMilli();
    	bench.execExecutionF = (bench.execExecution - bench.execPrepare)/1000.;
    }

    if (threadNumber == 0) {
    	//printf("%10.2f\t%10.5f\t%10.5f\n", dt * nSteps, (vmTimeSerie[0])[nCompVmTimeSerie*nKernelSteps-1], (vmTimeSerie[0])[nKernelSteps-1]);
    	//printf("%10.2f\t%10.5f\t%10.5f\n", dt * nSteps, (vmTimeSerie[1])[nCompVmTimeSerie*nKernelSteps-1], (vmTimeSerie[1])[nKernelSteps-1]);
    }

    // Used to print spike statistics in the end of the simulation
    if (threadNumber == 0)
    	sharedData->spkStat->printSpikeStatistics((const char *)"spikeGpu.dat", sharedData->totalTime, bench);

    // TODO: Free CUDA Memory
    if (threadNumber == 0) {
    	delete[] kernelInfo->nBlocksComm;
    	delete[] kernelInfo->nThreadsComm;
    }

    printf("Finished GPU execution.\n" );

    return 0;
}

