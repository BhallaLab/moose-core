/**********************************************************************
 ** This program is part of 'MOOSE', the
 ** Multiscale Object Oriented Simulation Environment.
 **   copyright (C) 2003-2011 Upinder S. Bhalla, Niraj Dudani and NCBS
 ** It is made available under the terms of the
 ** GNU Lesser General Public License version 2.1
 ** See the file COPYING.LIB for the full notice.
 **********************************************************************/

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
