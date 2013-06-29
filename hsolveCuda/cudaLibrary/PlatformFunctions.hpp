/*
 * PlatformFunctions.hpp
 *
 *  Created on: 17/06/2009
 *      Author: rcamargo
 */

#ifndef PLATFORMFUNCTIONS_HPP_
#define PLATFORMFUNCTIONS_HPP_

#include "Definitions.hpp"

#define NN_GPU 1
#define NN_CPU 2
#define NN_TEST 3

//#define CHK_COMM 2
//#define CPU_PROC 3

// connRead = Comm Kernel time
// connWait = global sync after Hines Kernel
// connWrite = Remaining communication cost, start from wait time and without comm kernel time
#define GPU_COMM_SIMPLE 1

// connRead = Comm Kernel Read time
// connWait = global sync after Hines Kernel
// connWrite = Comm Kernel Write time
#define GPU_COMM_DETAILED 2

#include <cstdio>

extern "C" {
uint64 gettimeInMilli();
}

class BenchConfig {

private:
	ucomp simCommMode;
	ucomp simProcMode;

public:
	ucomp printAllVmKernelFinish;
	ucomp printAllSpikeTimes;
	ucomp printSampleVms;

	ucomp assertResultsAll;
	ucomp verbose;

	ucomp gpuCommBenchMode;

	ucomp checkGpuComm;


	int checkProcMode (int type) {

		if (type == simProcMode)
			return simProcMode;
		else
			return 0;

	}

	int checkCommMode (int type) {

		if (type == simCommMode || simCommMode == NN_TEST)
			return simCommMode;
		else
			return 0;

	}

	int setMode (int typeProc, int typeComm) {

		if (typeProc == NN_CPU) {

			if (typeComm == NN_CPU) {
				simProcMode = typeProc;
				simCommMode = typeComm;
			}
			else {
				printf("Invalid simulation mode!!!\n");
				exit(-1);
			}

		}

		else if (typeProc == NN_GPU) {

			if (typeComm == NN_CPU || typeComm == NN_GPU || typeComm == NN_TEST) {
				simProcMode = typeProc;
				simCommMode = typeComm;
			}
			else {
				printf("Invalid simulation mode!!!\n");
				exit(-1);
			}


		}
		else {
			printf("Invalid simulation mode!!!\n");
			exit(-1);
		}

		return 0;
	}

};

struct BenchTimes {

	uint64 start;
	uint64 matrixSetup;
	uint64 execPrepare;
	uint64 execExecution;
	uint64 finish;

	ftype matrixSetupF;
	ftype execPrepareF;
	ftype execExecutionF;
	ftype finishF;

	//-----------------------------------------------------------
	// Used to benchmark the times for each piece of code
	//-----------------------------------------------------------

	uint64 kernelStart;
	uint64 kernelFinish;
	uint64 connRead;
	uint64 connWait;
	uint64 connWrite;
	uint64 mpiSpikeTransferStart;
	uint64 mpiSpikeTransferEnd;


	ftype totalHinesKernel;
	ftype totalConnRead;
	ftype totalConnWait;
	ftype totalConnWrite;

	//-----------------------------------
	ftype meanGenSpikes;
	ftype meanRecSpikes;
	ftype *meanGenSpikesType;
	ftype *meanRecSpikesType;
};

extern "C" {
extern struct BenchTimes bench;
extern struct BenchConfig benchConf;
}
#endif /* DEFINITIONS_HPP_ */
