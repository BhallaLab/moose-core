/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/
#include "header.h"
#include "SolverJunction.h"
#include "StoichPools.h"

extern const double NA;

const Cinfo* StoichPools::initCinfo()
{
	static DestFinfo addJunction( "addJunction",
			"Add a junction between the current solver and the one whose"
			" Id is passed in.",
			new EpFunc1< StoichPools, Id >( &StoichPools::addJunction )
	);

	static DestFinfo dropJunction( "dropJunction",
			"Drops a junction between the current solver and the one whose"
			" Id is passed in. Ignores if no junction.",
			new EpFunc1< StoichPools, Id >( &StoichPools::dropJunction )
	);

	static FieldElementFinfo< StoichPools, SolverJunction > junction(
		"junction",
		"Handles how solvers communicate with each other in case of "
		"diffusion, motors, or reaction.",
		SolverJunction::initCinfo(),
		&StoichPools::getJunction,
		&StoichPools::setNumJunctions,
		&StoichPools::getNumJunctions,
		2
	);

	static Finfo* stoichPoolsFinfos[] = {
		&addJunction,	// DestFinfo
		&dropJunction,	// DestFinfo
		&junction		// FieldElement
	};

	static string doc[] = 
	{
			"Name", "StoichPools",
			"Author", "Upinder S. Bhalla, 2012, NCBS",
			"Description", "Pure virtual base class for handling "
					"reaction pools. GslStoich is derived from this."
	};

	static Cinfo stoichPoolsCinfo(
		"StoichPools",
		Neutral::initCinfo(),
		stoichPoolsFinfos,
		sizeof( stoichPoolsFinfos ) / sizeof( Finfo* ),
		new ZeroSizeDinfo< int >(),
		doc, sizeof( doc ) / sizeof( string )
	);
	
	return &stoichPoolsCinfo;
}

static const Cinfo* stoichPoolsCinfo = StoichPools::initCinfo();

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

/*
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
*/

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

double* StoichPools::varSinit( unsigned int meshIndex )
{
	return &Sinit_[meshIndex][0];
}

unsigned int StoichPools::numMeshEntries() const
{
	return Sinit_.size();
}

unsigned int StoichPools::numPoolEntries( unsigned int i ) const
{
	if ( i >= Sinit_.size() )
			return 0;
	return Sinit_[i].size();
}

//////////////////////////////////////////////////////////////////////////
// Access functions for junctions
//////////////////////////////////////////////////////////////////////////
SolverJunction* StoichPools::getJunction( unsigned int i )
{
	static SolverJunction dummy;
	if ( i < junctions_.size() )
		return &junctions_[i];

	cout << "Warning: StoichPools::getJunction: Index: " << i << 
			" is out of range: " << junctions_.size() << endl;

	return &dummy;
}

void StoichPools::setNumJunctions( unsigned int v )
{
	cout << "Warning: StoichPools::setNumJunctions: Direct assignment "
		"of number not permitted, use addJunction instead.";
}	

unsigned int StoichPools::getNumJunctions() const
{
	return junctions_.size();
}

void StoichPools::handleJunction( unsigned int fieldIndex,
		vector< double > v )
{
	vHandleJunction( fieldIndex, v );
}

void StoichPools::addJunction( const Eref& e, const Qinfo* q, Id other )
{
	vAddJunction( e, q, other );
}


void StoichPools::dropJunction( const Eref& e, const Qinfo* q, Id other )
{
	vDropJunction( e, q, other );
}
