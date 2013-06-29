#ifndef NEURONINFOWRITER_H_
#define NEURONINFOWRITER_H_

#include "Definitions.hpp"
//#include "HinesStruct.hpp"
//#include "ThreadInfo.hpp"
//#include "SharedNeuronGpuData.hpp"
//#include "KernelInfo.hpp"

class NeuronInfoWriter {

private:
	struct KernelInfo *kernelInfo;
	struct ThreadInfo *tInfo;
	struct SharedNeuronGpuData *sharedData;

	ftype **vmTimeSerie;
	int vmTimeSerieMemSize;
	int nVmTimeSeries;

	FILE *outFile;
	FILE *vmKernelFile;
	FILE *resultFile;

	int *groupList;
	int *neuronList;

public:
	NeuronInfoWriter(struct ThreadInfo *tInfo);
	~NeuronInfoWriter();

	//void setMonitoredList( int nMonitored, int *groupList, int *neuronList );

    void writeVmToFile(int kStep);

    void updateSampleVm(int kStep);
    void writeSampleVm(int kStep);

    void writeResultsToFile(char mode, int nNeuronsTotal, int nComp, char* simType, struct BenchTimes & bench);
};

#endif /* NEURONINFOWRITER_H_ */
