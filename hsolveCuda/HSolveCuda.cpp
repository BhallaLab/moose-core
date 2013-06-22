/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**   copyright (C) 2003-2007 Upinder S. Bhalla, Niraj Dudani and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/
#include "HSolveCuda.h"

#include "cudaLibrary/CudaModule.h"

//#include <stdio.h>
#include <stdlib.h>

#include "header.h"
#include "ElementValueFinfo.h"
#include "../hsolve/HSolveStruct.h"
#include "../hsolve/HinesMatrix.h"
#include "../hsolve/HSolvePassive.h"
//#include "RateLookup.h"
//#include "HSolveActive.h"
//#include "../biophysics/Compartment.h"
//#include "ZombieCompartment.h"
//#include "../biophysics/CaConc.h"
//#include "ZombieCaConc.h"
//#include "../biophysics/HHGate.h"
//#include "../biophysics/ChanBase.h"
//#include "../biophysics/HHChannel.h"
//#include "ZombieHHChannel.h"


HSolveCuda::HSolveCuda(const unsigned int cellNumber, const unsigned int nComp)
{
	__nCells = cellNumber;
	__nComps = nComp;
	__hostMatrix = (float *) malloc(__nComps *__nComps * sizeof(float) * cellNumber);
	V = (float *) malloc(__nComps * sizeof(float) * cellNumber);
	Cm = (float *) malloc(__nComps * sizeof(float) * cellNumber);
	Rm = (float *) malloc(__nComps * sizeof(float) * cellNumber);
	Ra = (float *) malloc(__nComps * sizeof(float) * cellNumber);
	Em = (float *) malloc(__nComps * sizeof(float) * cellNumber);
	B = (float *) malloc(__nComps * sizeof(float) * cellNumber);
	__dt = 0;
	//TODO Saeed: Make everything zero with parallel approaches
	std::memset(__hostMatrix,__nComps *__nComps * sizeof(float) * cellNumber,0);
}

HSolveCuda::~HSolveCuda()
{
	free(__hostMatrix);
	free(V);
	free(Cm);
	free(Rm);
	free(Ra);
	free(Em);
	free(B);
}

void HSolveCuda::TestCudaAbility()
{
	cudaModule_test();
}

void HSolveCuda::Setup(const double dt)
{
	//Initialize device and Send data
	cudaModule_setup(
			__nCells,
			__nComps,
			__hostMatrix,
			V,
			Cm,
			Em,
			Rm,
			Ra
			);


//	__nComp = matrix.size();
//	__hostMatrix = (float *) malloc(__nComp*__nComp * sizeof(float));
//	__dt = dt;
//	for ( unsigned int i = 0; i < __nComp; i++)
//	{
//		__V[i] = V[i];
//		__Em[i] = Em[i];
//		__Cm[i] = tree[i].Cm;
//		__Rm[i] = tree[i].Rm;
//		for ( unsigned int j = 0; j < __nComp; j++ )
//			__hostMatrix[i*__nComp+j]=matrix[i][j];
//	}
}

void HSolveCuda::UpdateMatrix()
{
	cudaModule_updateMatrix(__dt);
}
void HSolveCuda::ForwardElimination()
{
	cudaMosule_forwardElimination();
}
void HSolveCuda::BackwardSubstitute()
{
	cudaMosule_backwardSubstitute();
}

// Get functions
int HSolveCuda::Read_B()
{
	return cudaModule_getB(B);
}

///////////////////////////////////////////////////
// Helper function definitions
///////////////////////////////////////////////////

void HSolveCuda::AddFullMatrix(
	const unsigned int cellNumber,
	const vector< TreeNodeStruct >& tree,
	double dt)
{
	__dt = dt;
	unsigned int baseIndex = cellNumber*__nComps*__nComps;
	unsigned int size = tree.size();
	if(size != __nComps)
		cerr << "Saeed: wrong dimension size"<<flush;
	/*
	 * Some convenience variables
	 */
	vector< double > CmByDt;
	vector< double > Ga;
	for ( unsigned int i = 0; i < tree.size(); i++ ) {
		CmByDt.push_back( tree[ i ].Cm / ( dt / 2.0 ) );
		Ga.push_back( 2.0 / tree[ i ].Ra );
	}

	/* Each entry in 'coupled' is a list of electrically coupled compartments.
	 * These compartments could be linked at junctions, or even in linear segments
	 * of the cell.
	 */
	vector< vector< unsigned int > > coupled;
	for ( unsigned int i = 0; i < tree.size(); i++ )
		if ( tree[ i ].children.size() >= 1 ) {
			coupled.push_back( tree[ i ].children );
			coupled.back().push_back( i );
		}


	// Setting diagonal elements
	for ( unsigned int i = 0; i < size; i++ )
		__hostMatrix[ baseIndex+ i* __nComps + i ] = CmByDt[ i ] + 1.0 / tree[ i ].Rm;

	double gi;
	vector< vector< unsigned int > >::iterator group;
	vector< unsigned int >::iterator ic;
	for ( group = coupled.begin(); group != coupled.end(); ++group ) {
		double gsum = 0.0;

		for ( ic = group->begin(); ic != group->end(); ++ic )
			gsum += Ga[ *ic ];

		for ( ic = group->begin(); ic != group->end(); ++ic ) {
			gi = Ga[ *ic ];

			__hostMatrix[ baseIndex+ (*ic) * __nComps + (*ic) ] +=
					gi * ( 1.0 - gi / gsum );
		}
	}

	// Setting off-diagonal elements
	double gij;
	vector< unsigned int >::iterator jc;
	for ( group = coupled.begin(); group != coupled.end(); ++group ) {
		double gsum = 0.0;

		for ( ic = group->begin(); ic != group->end(); ++ic )
			gsum += Ga[ *ic ];

		for ( ic = group->begin(); ic != group->end() - 1; ++ic ) {
			for ( jc = ic + 1; jc != group->end(); ++jc ) {
				gij = Ga[ *ic ] * Ga[ *jc ] / gsum;

				__hostMatrix[ baseIndex+ (*ic) * __nComps + (*jc) ] = -gij;
				__hostMatrix[ baseIndex+ (*jc) * __nComps + (*ic) ] = -gij;
			}
		}
	}
}





