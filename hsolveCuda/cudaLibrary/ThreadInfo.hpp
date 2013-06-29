#ifndef THREADINFO_HPP
#define THREADINFO_HPP

struct ThreadInfo{
	struct SharedNeuronGpuData *sharedData;	// Shared among the threads
	int *nNeurons;						// Shared among the threads // length = totalTypes
	int *nComp;							// Shared among the threads

	int kStep;

	int nTypes;
	int totalTypes;
	int *typeProcess; // The rank of the process assigned to that type
	int *nNeuronsTotalType; // length = nTypes

	int startTypeThread;
	int endTypeThread;
	int threadNumber;

    struct cudaDeviceProp *prop;
	int deviceNumber;
};


#endif
