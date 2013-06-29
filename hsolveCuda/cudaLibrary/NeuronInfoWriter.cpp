#include <cstdio>
#include "NeuronInfoWriter.hpp"
#include "SynapticData.hpp"
#include "KernelInfo.hpp"
#include "ThreadInfo.hpp"
#include "HinesMatrix.hpp"
#include "SharedNeuronGpuData.hpp"
#include "HinesStruct.hpp"

#include <cuda.h>
#include <cuda_runtime_api.h>

NeuronInfoWriter::NeuronInfoWriter(ThreadInfo *tInfo) {

	this->tInfo = tInfo;
	this->sharedData = tInfo->sharedData;
	this->kernelInfo = tInfo->sharedData->kernelInfo;

    char buf[20];
    sprintf(buf, "%s", "results0.dat");
    this->resultFile = fopen(buf, "a");

    sprintf(buf, "%s", "sampleVm0.dat");
    this->outFile = fopen(buf, "w");

    sprintf(buf, "%s", "vmKernel0.dat");
    this->vmKernelFile = fopen(buf, "w");


    this->nVmTimeSeries = 4;
    int nCompVmTimeSerie = sharedData->matrixList[tInfo->startTypeThread][0].nComp;
    this->vmTimeSerie = (ftype**)(((malloc(sizeof (ftype*) * this->nVmTimeSeries))));

    this->vmTimeSerieMemSize = sizeof (ftype) * (nCompVmTimeSerie * kernelInfo->nKernelSteps);
    for(int k = 0;k < nVmTimeSeries;k++)
    	this->vmTimeSerie[k] = (ftype*)(((malloc(vmTimeSerieMemSize))));

    groupList  = (int *)malloc(sizeof(int)*nVmTimeSeries);
    neuronList = (int *)malloc(sizeof(int)*nVmTimeSeries);
    neuronList[0] = 0;
    neuronList[1] = 1;
    neuronList[2] = 2;
    neuronList[3] = 3;

    groupList[0]  = tInfo->startTypeThread;
    groupList[1]  = tInfo->startTypeThread;

    if (tInfo->endTypeThread - tInfo->startTypeThread > 1) {
    	groupList[2]  = tInfo->startTypeThread+1;
    	groupList[3]  = tInfo->startTypeThread+1;
    }
    else {
        groupList[2]  = tInfo->startTypeThread;
        groupList[3]  = tInfo->startTypeThread;
    }



}

NeuronInfoWriter::~NeuronInfoWriter () {

	fclose(outFile);
	fclose(vmKernelFile);

	for(int k = 0;k < nVmTimeSeries; k++)
		free( vmTimeSerie[k] );

	free (vmTimeSerie);

}

//void NeuronInfoWriter::setMonitoredList( int nMonitored, int *groupList, int *neuronList ) {
//}

void NeuronInfoWriter::writeVmToFile(int kStep) {

    for(int type = 0; type < tInfo->totalTypes;type++){
        fprintf(vmKernelFile, "dt=%-10.2f\ttype=%d\t", sharedData->dt * (kStep + kernelInfo->nKernelSteps), type);
        for(int n = 0;n < tInfo->nNeurons[type];n++)
            fprintf(vmKernelFile, "%10.2f\t", sharedData->synData->vmListHost[type][n]);

        fprintf(vmKernelFile, "\n");
    }
}

void NeuronInfoWriter::updateSampleVm(int kStep) {

	int pos = kStep % kernelInfo->nKernelSteps;
	HinesMatrix **matrixList = tInfo->sharedData->matrixList;
    for(int k = 0; k < nVmTimeSeries;k++)
    	vmTimeSerie[k][pos] = matrixList[ groupList[k] ][ neuronList[k] ].vmList[0];

}

void NeuronInfoWriter::writeSampleVm(int kStep)
{
    if(benchConf.verbose == 1)
        printf("Writing Sample Vms thread=%d\n", tInfo->threadNumber);

    for(int k = 0; k < nVmTimeSeries; k++)
    	if(tInfo->startTypeThread <= groupList[k] && groupList[k] < tInfo->endTypeThread)
    		cudaMemcpy(vmTimeSerie[k], sharedData->hList[ groupList[k] ][ neuronList[k] ].vmTimeSerie,
    				vmTimeSerieMemSize, cudaMemcpyDeviceToHost);

	for(int i = kStep;i < kStep + kernelInfo->nKernelSteps; i++)
		fprintf(outFile, "%10.2f\t%10.2f\t%10.2f\t%10.2f\t%10.2f\n", sharedData->dt * (i + 1),
			vmTimeSerie[0][i-kStep], vmTimeSerie[1][i-kStep], vmTimeSerie[2][i - kStep], vmTimeSerie[3][i - kStep]);
}

void NeuronInfoWriter::writeResultsToFile(char mode, int nNeuronsTotal, int nComp, char* simType, BenchTimes & bench) {

	printf ("Setup=%-10.3f Prepare=%-10.3f Execution=%-10.3f Total=%-10.3f\n", bench.matrixSetupF, bench.execPrepareF, bench.execExecutionF, bench.finishF);
	printf ("HinesKernel=%-10.3f ConnRead=%-10.3f ConnWait=%-10.3f ConnWrite=%-10.3f\n",
			bench.totalHinesKernel, bench.totalConnRead, bench.totalConnWait, bench.totalConnWrite);
	printf ("%f %f %f\n", tInfo->sharedData->inputSpikeRate, tInfo->sharedData->pyrPyrConnRatio, tInfo->sharedData->pyrInhConnRatio);

	fprintf (resultFile, "mode=%c neurons=%-6d types=%-2d comp=%-2d threads=%d ftype=%lu simtype=%s\n",
			mode, nNeuronsTotal, tInfo->totalTypes, nComp, sharedData->nThreadsCpu, sizeof(ftype), simType);
	fprintf (resultFile, "meanGenSpikes[T|P|I|B]=[%-10.5f|%-10.5f|%-10.5f|%-10.5f]\n",
			bench.meanGenSpikes, bench.meanGenSpikesType[PYRAMIDAL_CELL], bench.meanGenSpikesType[INHIBITORY_CELL], bench.meanGenSpikesType[BASKET_CELL]);
	fprintf (resultFile, "meanRecSpikes[T|P|I]=[%-10.5f|%-10.5f|%-10.5f|%-10.5f] \n",
			bench.meanRecSpikes, bench.meanRecSpikesType[PYRAMIDAL_CELL], bench.meanRecSpikesType[INHIBITORY_CELL], bench.meanRecSpikesType[BASKET_CELL]);
	fprintf (resultFile, "inpRate=%-5.3f pyrRatio=%-5.3f inhRatio=%-5.3f nKernelSteps=%d\n",
			tInfo->sharedData->inputSpikeRate, tInfo->sharedData->pyrPyrConnRatio,
			tInfo->sharedData->pyrInhConnRatio, kernelInfo->nKernelSteps);
	fprintf (resultFile, "Setup=%-10.3f Prepare=%-10.3f Execution=%-10.3f Total=%-10.3f\n",
			bench.matrixSetupF, bench.execPrepareF, bench.execExecutionF, bench.finishF);
	fprintf (resultFile, "HinesKernel=%-10.3f ConnRead=%-10.3f ConnWait=%-10.3f ConnWrite=%-10.3f\n",
			bench.totalHinesKernel, bench.totalConnRead, bench.totalConnWait, bench.totalConnWrite);
	fprintf (resultFile, "#------------------------------------------------------------------------------\n");

	fclose(outFile);
}

