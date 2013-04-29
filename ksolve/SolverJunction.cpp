/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2012 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
#include "../mesh/VoxelJunction.h"
#include "SolverJunction.h"
#include "SolverBase.h"
#include "UpFunc.h"

SrcFinfo1< vector< double > >* junctionPoolDeltaFinfo()
{
	static SrcFinfo1< vector< double > > junctionPoolDelta(
		"junctionPoolDelta",
		"Sends out vector of all mol # changes going across junction."
	);
	return &junctionPoolDelta;
}

SrcFinfo1< vector< double > >* junctionPoolNumFinfo()
{
	static SrcFinfo1< vector< double > > junctionPoolNum(
		"junctionPoolNum",
		"Sends out vector of all mol #s needed to compute junction rates."
	);
	return &junctionPoolNum;
}

const Cinfo* SolverJunction::initCinfo()
{
	//////////////////////////////////////////////////////////////////
	// Fields
	//////////////////////////////////////////////////////////////////
	
	static ReadOnlyValueFinfo< SolverJunction, unsigned int > numReacs(
	   	"numReacs",
		"Number of cross-compartment reactions on this Junction",
		&SolverJunction::getNumReacs
	);

	static ReadOnlyValueFinfo< SolverJunction, unsigned int > 
		numDiffMols( 
		"numDiffMols",
		"Number of molecule species diffusing across this Junction",
		&SolverJunction::getNumDiffMols
	);

	static ReadOnlyValueFinfo< SolverJunction, unsigned int > 
		numMeshEntries( 
		"numMeshEntries",
		"Number of voxels (mesh entries) handled by Junction",
		&SolverJunction::getNumMeshIndex
	);

	static ReadOnlyValueFinfo< SolverJunction, Id > myCompartment( 
		"myCompartment",
		"Id of compartment containing this Junction. ",
		&SolverJunction::getMyCompartment
	);

	static ReadOnlyValueFinfo< SolverJunction, Id > otherCompartment( 
		"otherCompartment",
		"Id of compartment on other side of this Junction. ",
		&SolverJunction::getOtherCompartment
	);
	//////////////////////////////////////////////////////////////////
	// DestFinfos
	//////////////////////////////////////////////////////////////////
	static DestFinfo handleJunctionPoolDelta( "handleJunctionPoolDelta",
		"Handles vector of doubles with pool num changes that arrive at"
		" the Junction, by redirecting up to"
	   	" parent StoichPools object",
		new UpFunc1< SolverBase, vector< double > >( 
				&SolverBase::handleJunctionPoolDelta ) );
	static DestFinfo handleJunctionPoolNum( "handleJunctionPoolNum",
		"Handles vector of doubles specifying pool num, that arrive at"
		" the Junction, by redirecting up to"
	   	" parent StoichPools object",
		new UpFunc1< SolverBase, vector< double > >( 
				&SolverBase::handleJunctionPoolNum ) );

	//////////////////////////////////////////////////////////////////
	// Shared Finfos
	//////////////////////////////////////////////////////////////////
	static Finfo* symJunctionShared[] = {
			&handleJunctionPoolNum,
			junctionPoolNumFinfo(),
	};

	static SharedFinfo symJunction( "symJunction",
		"Symmetric shared message between SolverJunctions to handle "
		"cross-solver reactions and diffusion. This variant sends only "
		"pool mol#s, and is symmetric.",
		symJunctionShared,
		sizeof( symJunctionShared ) / sizeof( const Finfo* )
	);

	static Finfo* masterJunctionShared[] = {
			&handleJunctionPoolNum,
			junctionPoolDeltaFinfo(),
	};
	static SharedFinfo masterJunction( "masterJunction",
		"Shared message between SolverJunctions to handle cross-solver "
		"reactions and diffusion. This sends the change in pool #, "
		"of abutting voxels, and receives the pool# of the same abutting "
		"voxels. Thus it operates on the solver that is doing the "
		"diffusion calculations. This will typically be the solver that "
		"operates at a finer level of detail. The order of detail is "
		"Smoldyn > Gillespie > deterministic. "
		"For two identical solvers we would typically have one with the "
		"finer grid size become the master Junction. ",
		masterJunctionShared,
		sizeof( masterJunctionShared ) / sizeof( const Finfo* )
	);

	static Finfo* followerJunctionShared[] = {
			&handleJunctionPoolDelta,
			junctionPoolNumFinfo(),
	};
	static SharedFinfo followerJunction( "followerJunction",
		"Shared message between SolverJunctions to handle cross-solver "
		"reactions and diffusion. This sends the pool #, "
		"of its boundary voxels, and receives back changes in the pool# "
	    "of the same boundary voxels "
		"voxels. Thus it operates on the solver that is just tracking the "
		"diffusion calculations that the other (master) solver is doing",
		followerJunctionShared,
		sizeof( followerJunctionShared ) / sizeof( const Finfo* )
	);

	static Finfo* solverJunctionFinfos[] = {
		&numReacs,				// ReadOnly Fields
		&numDiffMols,			// ReadOnly Fields
		&numMeshEntries,		// ReadOnly Fields
		&myCompartment,			// ReadOnly Fields
		&otherCompartment,		// ReadOnly Fields
		&symJunction, 			// SharedFinfo
		&masterJunction, 		// SharedFinfo
		&followerJunction, 		// SharedFinfo
	};

	static Cinfo solverJunctionCinfo (
		"SolverJunction",
		Neutral::initCinfo(),
		solverJunctionFinfos,
		sizeof( solverJunctionFinfos ) / sizeof ( Finfo* ),
		new Dinfo< SolverJunction >()
	);

	return &solverJunctionCinfo;
}

static const Cinfo* solverJunctionCinfo = SolverJunction::initCinfo();

SolverJunction::SolverJunction()
{
	;
}

SolverJunction::~SolverJunction()
{
	;
}

/////////////////////////////////////////////////////////////////////////
// Fields
/////////////////////////////////////////////////////////////////////////

unsigned int SolverJunction::getNumReacs() const
{
		return reacTerms_.size();
}

unsigned int SolverJunction::getNumDiffMols() const
{
		return diffTerms_.size();
}
unsigned int SolverJunction::getNumMeshIndex() const
{
		return sendMeshIndex_.size();
}

Id SolverJunction::getOtherCompartment() const
{
		return otherCompartment_;
}

Id SolverJunction::getMyCompartment() const
{
		return myCompartment_;
}

/////////////////////////////////////////////////////////////////////////
// Utility functions
/////////////////////////////////////////////////////////////////////////
const vector< unsigned int >& SolverJunction::reacTerms() const
{
	return reacTerms_;
}


const vector< unsigned int >& SolverJunction::diffTerms() const
{
	return diffTerms_;
}

const vector< VoxelJunction >& SolverJunction::meshMap() const
{
	return targetMeshIndices_;
}

const vector< unsigned int >& SolverJunction::remoteReacPools() const
{
	return remoteReacPools_;
}

const vector< unsigned int >& SolverJunction::localReacPools() const
{
	return localReacPools_;
}

const vector< unsigned int >& SolverJunction::sendPoolIndex() const
{
	return sendPoolIndex_;
}
const vector< unsigned int >& SolverJunction::sendMeshIndex() const
{
	return sendMeshIndex_;
}
const vector< unsigned int >& SolverJunction::abutPoolIndex() const
{
	return abutPoolIndex_;
}
const vector< unsigned int >& SolverJunction::abutMeshIndex() const
{
	return abutMeshIndex_;
}

void SolverJunction::incrementTargets( 
				vector< vector< double > >& y,
				const vector< double >& v ) const
{
	typedef vector< pair< unsigned int, unsigned int> >::const_iterator VPI;

	unsigned int numReacTerms = localReacPools_.size() * 
			sendMeshIndex_.size();
	assert( v.size() == numReacTerms + 
		sendMeshIndex_.size() * diffTerms_.size() );

	/*
	// handle the chemical terms
	for ( vector< unsigned int >::const_iterator i = 
			meshIndex_.begin(); i != meshIndex_.end(); ++i ) {
		for ( VPI j = targetMols_.begin(); j != targetMols_.end(); ++j ) {
			y[ *i][ j->second ] += v[ *i * meshIndex_.size() + j->first ];
		}
	}
	*/

	vector< double >::const_iterator iv = v.begin();

	// Handle the chemical and diffusion terms.
	// Note this is the sendMeshIndex: for the core voxels on this solver,
	// whose values have to be incremented based on what happened elsewhere.
	for ( vector< unsigned int >::const_iterator i = 
		sendMeshIndex_.begin(); i != sendMeshIndex_.end(); ++i ) {
		for ( vector< unsigned int >::const_iterator j = 
			localReacPools_.begin(); j != localReacPools_.end(); ++j ) {
			y[ *i ][ *j ] += *iv++;
		}
	}

	assert( iv == v.begin() + numReacTerms);
	for ( vector< unsigned int >::const_iterator i = 
		sendMeshIndex_.begin(); i != sendMeshIndex_.end(); ++i ) {
		for ( vector< unsigned int >::const_iterator j = 
			diffTerms_.begin(); j != diffTerms_.end(); ++j ) {
			y[ *i ][ *j ] += *iv++;
		}
	}
}

void SolverJunction::setDiffTerms( const vector< unsigned int >& diffTerms )
{
	diffTerms_ = diffTerms;
}

void SolverJunction::setLocalReacPools( const vector< unsigned int >& pools)
{
	localReacPools_ = pools;
}

void SolverJunction::setRemoteReacPools( const vector< unsigned int >& pools)
{
	remoteReacPools_ = pools;
}

void SolverJunction::setMeshMap( const vector< VoxelJunction >& meshMap )
{
	targetMeshIndices_ = meshMap;
}


void SolverJunction::setSendPools( 
	const vector< unsigned int >& meshIndex,
	const vector< unsigned int >& poolIndex )
{
	sendMeshIndex_ = meshIndex;
	sendPoolIndex_ = poolIndex;
}

void SolverJunction::setAbutPools( 
	const vector< unsigned int >& meshIndex,
	const vector< unsigned int >& poolIndex)
{
	if ( poolIndex.size() > 0 ) {
		abutMeshIndex_ = meshIndex;
	} else {
		abutMeshIndex_.resize( 0 );
	}
	abutPoolIndex_ = poolIndex;
}

void SolverJunction::setCompartments( Id myCompt, Id otherCompt )
{
	myCompartment_ = myCompt;
	otherCompartment_ = otherCompt;
}
