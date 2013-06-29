#ifndef KERNELINFO_HPP
#define KERNELINFO_HPP

struct KernelInfo{

	int nKernelSteps; // Number of integration steps performed on each kernel call

	int *nBlocksComm;  // Defined only for the types of the current process
	int *nThreadsComm; // Defined only for the types of the current process

	int *nBlocksProc;  // Defined only for the types of the current process

    int maxThreadsProc;
    int maxThreadsComm;
    int sharedMemSizeProc;
    int sharedMemSizeComm;

};

#endif
