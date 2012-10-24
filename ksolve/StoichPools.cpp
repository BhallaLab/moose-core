/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/
#include <vector>
using namespace std;

#include <cassert>
#include "StoichPools.h"

extern const double NA;

//////////////////////////////////////////////////////////////
// Class definitions
//////////////////////////////////////////////////////////////

StoichPools::StoichPools()
	: 
		S_(1),
		Sinit_(1),
		localMeshEntries_(1, 0)
{;}

StoichPools::~StoichPools()
{;}

/// Using the computed array sizes, now allocate space for them.
void StoichPools::resizeArrays( unsigned int totNumPools )
{
	S_.resize( 1 );
	Sinit_.resize( 1 );
	S_[0].resize( totNumPools, 0.0 );
	Sinit_[0].resize( totNumPools, 0.0);
}

void StoichPools::meshSplit( 
				vector< double > initConcs,  // in milliMolar
				vector< double > vols,		// in m^3
				vector< unsigned int > localEntryList )
{
	assert( S_[0].size() == initConcs.size() );
	assert( vols.size() == localEntryList.size() );
	unsigned int numPools = initConcs.size();
	unsigned int numLocalVoxels = vols.size();

	S_.resize( numLocalVoxels );
	Sinit_.resize( numLocalVoxels );
	for ( unsigned int i = 0; i < numLocalVoxels; ++i ) {
		double v = vols[i] * NA;
		S_[i].resize( numPools, 0 );
		Sinit_[i].resize( numPools, 0 );
		for ( unsigned int j = 0; j < initConcs.size(); ++j ) 
			S_[i][j] = Sinit_[i][j] = initConcs[j] * v;
	}
	// localMeshEntries_ = localEntryList;
}

void StoichPools::innerSetN( unsigned int meshIndex, 
				unsigned int poolIndex, double v )
{
	assert( poolIndex < S_[meshIndex].size() );
	S_[ meshIndex ][ poolIndex ] = v;
}

void StoichPools::innerSetNinit( unsigned int meshIndex, 
				unsigned int poolIndex, double v )
{
	Sinit_[ meshIndex ][ poolIndex ] = v;
}

const double* StoichPools::S( unsigned int meshIndex ) const
{
	return &S_[meshIndex][0];
}

double* StoichPools::varS( unsigned int meshIndex )
{
	return &S_[meshIndex][0];
}

const double* StoichPools::Sinit( unsigned int meshIndex ) const
{
	return &Sinit_[meshIndex][0];
}

