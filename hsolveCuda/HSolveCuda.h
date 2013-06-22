
#ifndef _HSolveCuda_H
#define _HSolveCuda_H

#include <vector>
#include <map>
using namespace std;
#include <../hsolve/HinesMatrix.h>
#include "cudaLibrary/CudaModule.h"

/**
 *
 */
class HSolveCuda
{
public:
	void TestCudaAbility();
	HSolveCuda(const unsigned int cellNumber, const unsigned int __nComp);
	~HSolveCuda();
	void Setup(const double dt);
	void AddFullMatrix(
			const unsigned int cellNumber,
		const vector< TreeNodeStruct >& tree,
		double dt);

	void UpdateMatrix( );
	void ForwardElimination();
	void BackwardSubstitute();
	int Read_B();

	float * V;
	float * Cm;
	float * Em;
	float * Rm;
	float * Ra;
	float * B;

private:
	float * __hostMatrix;
	unsigned int __nComps;
	unsigned int __nCells;
	float __dt;
};

#endif // _HSolveCuda_H
