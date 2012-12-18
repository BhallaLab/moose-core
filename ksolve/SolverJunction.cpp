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
#include "StoichPools.h"
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

	static ReadOnlyValueFinfo< SolverJunction, Id > otherCompartment( 
		"otherCompartment",
		"Id of compartment on other side of this Junction. "
		"Readily obtained by message traversal, just a utility field.",
		&SolverJunction::getOtherCompartment
	);
	//////////////////////////////////////////////////////////////////
	// DestFinfos
	//////////////////////////////////////////////////////////////////
	static DestFinfo handleJunctionPoolDelta( "handleJunctionPoolDelta",
		"Handles vector of doubles with pool num changes that arrive at"
		" the Junction, by redirecting up to"
	   	" parent StoichPools object",
		new UpFunc1< StoichPools, vector< double > >( 
				&StoichPools::handleJunctionPoolDelta ) );
	static DestFinfo handleJunctionPoolNum( "handleJunctionPoolNum",
		"Handles vector of doubles specifying pool num, that arrive at"
		" the Junction, by redirecting up to"
	   	" parent StoichPools object",
		new UpFunc1< StoichPools, vector< double > >( 
				&StoichPools::handleJunctionPoolNum ) );

	//////////////////////////////////////////////////////////////////
	// Shared Finfos
	//////////////////////////////////////////////////////////////////
	static Finfo* junctionShared[] = {
			&handleJunctionPoolNum,
			&handleJunctionPoolDelta,
			junctionPoolNumFinfo(),
			junctionPoolDeltaFinfo(),
	};

	static SharedFinfo junction( "junction",
		"Shared message between SolverJunctions to handle cross-solver "
		"reactions and diffusion.",
		junctionShared,
		sizeof( junctionShared ) / sizeof( const Finfo* )
	);

	static Finfo* solverJunctionFinfos[] = {
		&numReacs,				// ReadOnly Fields
		&numDiffMols,			// ReadOnly Fields
		&numMeshEntries,		// ReadOnly Fields
		&otherCompartment,		// ReadOnly Fields
		&junction, 				// SharedFinfo
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
		return meshIndex_.size();
}

Id SolverJunction::getOtherCompartment() const
{
		return Id(); // Dummy for now.
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

const vector< unsigned int >& SolverJunction::meshIndex() const
{
	return meshIndex_;
}

const vector< VoxelJunction >& SolverJunction::meshMap() const
{
	return targetMeshIndices_;
}

const vector< unsigned int >& SolverJunction::sendPoolIndex() const
{
	return sendPoolIndex_;
}
const vector< unsigned int >& SolverJunction::sendMeshIndex() const
{
	return sendMeshIndex_;
}
const vector< unsigned int >& SolverJunction::recvPoolIndex() const
{
	return recvPoolIndex_;
}
const vector< unsigned int >& SolverJunction::recvMeshIndex() const
{
	return recvMeshIndex_;
}

void SolverJunction::incrementTargets( 
				vector< vector< double > >& y,
				const vector< double >& v ) const
{
	typedef vector< pair< unsigned int, unsigned int> >::const_iterator VPI;

	unsigned int numReacTerms = targetMols_.size() * meshIndex_.size();
	assert( v.size() == numReacTerms + 
		sendMeshIndex_.size() * diffTerms_.size() );

	// handle the chemical terms
	for ( vector< unsigned int >::const_iterator i = 
			meshIndex_.begin(); i != meshIndex_.end(); ++i ) {
		for ( VPI j = targetMols_.begin(); j != targetMols_.end(); ++j ) {
			y[ *i][ j->second ] += v[ *i * meshIndex_.size() + j->first ];
		}
	}

	vector< double >::const_iterator iv = v.begin() + numReacTerms;

	// Handle the diffusion terms.
	/*
	 * This no longer applies, because we transfer the delta for each pool,
	 * not for each possible voxelJunction.
	for ( vector< VoxelJunction >::const_iterator 
					i = targetMeshIndices_.begin(); 
					i != targetMeshIndices_.end(); ++i ) {
		for ( unsigned int j = 0; j < diffTerms_.size(); ++j ) {
			y[ i->second ][ diffTerms_[j] ] += 
				v[ i->first * diffTerms_.size() + j ];
		}
	}
	*/
	/*
	for ( unsigned int i = 0; i < recvMeshIndex_.size(); ++i ) {
		for ( unsigned int j = 0; j < diffTerms_.size(); ++j ) {
			y[ recvMeshIndex_[i] ][ diffTerms_[j] ] += 
				v[ i * diffTerms_.size() + j ];
		}
	}
	*/

	// Note this is the sendMeshIndex: for the core voxels on this solver,
	// whose values have to be incremented based on what happened elsewhere.
	for ( vector< unsigned int >::const_iterator i = 
		sendMeshIndex_.begin(); i != sendMeshIndex_.end(); ++i ) {
		for ( vector< unsigned int >::const_iterator j = 
			diffTerms_.begin(); j != diffTerms_.end(); ++j ) {
			y[ *i ][ *j ] += *iv++;
		}
	}
}

void SolverJunction::setReacTerms( const vector< unsigned int >& reacTerms,
	const vector< pair< unsigned int, unsigned int > >& poolMap )
{
	reacTerms_ = reacTerms;
	targetMols_ = poolMap;
}

void SolverJunction::setDiffTerms( const vector< unsigned int >& diffTerms )
{
	diffTerms_ = diffTerms;
}

void SolverJunction::setMeshIndex( const vector< unsigned int >& meshIndex,
	const vector< VoxelJunction >& meshMap )
{
	meshIndex_ = meshIndex;
	targetMeshIndices_ = meshMap;
}


void SolverJunction::setSendPools( 
	const vector< unsigned int >& meshIndex,
	const vector< unsigned int >& poolIndex )
{
	sendMeshIndex_ = meshIndex;
	sendPoolIndex_ = poolIndex;
}

void SolverJunction::setRecvPools( 
	const vector< unsigned int >& meshIndex,
	const vector< unsigned int >& poolIndex)
{
	recvMeshIndex_ = meshIndex;
	recvPoolIndex_ = poolIndex;
}
