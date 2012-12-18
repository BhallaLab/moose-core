/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/
#include "header.h"
#include "../mesh/VoxelJunction.h"
#include "SolverJunction.h"
#include "StoichPools.h"
#include "../shell/Shell.h"

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
		16
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

// This function has to be updated to deal with changed boundaries. Ugh.
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
	localMeshEntries_ = localEntryList;
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

double* StoichPools::varSinit( unsigned int meshIndex )
{
	return &Sinit_[meshIndex][0];
}

unsigned int StoichPools::numMeshEntries() const
{
	return localMeshEntries_.size();
}

unsigned int StoichPools::numAllMeshEntries() const
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

void StoichPools::handleJunctionPoolDelta( unsigned int fieldIndex,
		vector< double > v )
{
	vHandleJunctionPoolDelta( fieldIndex, v );
}

void StoichPools::handleJunctionPoolNum( unsigned int fieldIndex,
		vector< double > v )
{
	vHandleJunctionPoolNum( fieldIndex, v );
}

bool validateJunction( Id me, Id other )
{
	// Check that types match
	if ( !other.element()->cinfo()->isA( "StoichPools" ) ) {
		cout << "Warning: StoichPools::validateJunction: Other Id '" <<
				other.path() << " is not a StoichPool\n";
		return false;
	}
	// Check if junction already exists.
	Id myJunction( me.value() + 1);
	Id otherJunction( other.value() + 1);
	vector< Id > ret;
	myJunction.element()->getNeighbours( ret, junctionPoolDeltaFinfo() );
	if ( find( ret.begin(), ret.end(), otherJunction ) != ret.end() ) {
		cout << "Warning: StoichPools::validateJunction: junction " <<
		" already present from " << 
		me.path() << " to " << other.path() << endl;
		return false;
	}
	return true;
}

void StoichPools::findDiffusionTerms( 
				const StoichPools* otherSP,
				vector< unsigned int >& selfTerms,
				vector< unsigned int >& otherTerms
	) const
{
	selfTerms.resize( 0 );
	otherTerms.resize( 0 );
	map< string, unsigned int > selfDiffTerms;
	this->vBuildDiffTerms( selfDiffTerms );
	map< string, unsigned int > otherDiffTerms;
	otherSP->vBuildDiffTerms( otherDiffTerms );
	map< string, unsigned int >::iterator j;
	for ( map< string, unsigned int >::iterator i = selfDiffTerms.begin();
					i != selfDiffTerms.end(); ++i )
	{
		j = otherDiffTerms.find( i->first );
		if ( j != otherDiffTerms.end() ) {
			selfTerms.push_back( i->second );
			otherTerms.push_back( j->second );
		}
	}
}

void StoichPools::innerConnectJunctions( 
			Id me, Id other, StoichPools* otherSP )
{
	Shell* shell = reinterpret_cast< Shell* >( Id().eref().data() );
	Id myJunction( me.value() + 1);
	Id otherJunction( other.value() + 1);
	// Set up message
	ObjId myOid( myJunction, getNumJunctions() );
	ObjId otherOid( otherJunction, otherSP->getNumJunctions() );
	MsgId mid = shell->doAddMsg( "single", myOid, "junction", 
					otherOid, "junction" );
	assert( mid != Msg::bad );

	// Make junction entries
	junctions_.resize( junctions_.size() + 1 );
	otherSP->junctions_.resize( otherSP->junctions_.size() + 1 );
}

void StoichPools::expandSforDiffusion( 
		const vector< unsigned int > & otherMeshIndex,
		const vector< unsigned int > & selfDiffPoolIndex,
		SolverJunction& j )
{
	vector< unsigned int > recvMeshIndex( otherMeshIndex.size(), 0 );
	for ( unsigned int i = 0; i < otherMeshIndex.size(); ++i )
		recvMeshIndex[i] = i + S_.size();
	vector< double > temp( S_[0].size(), 0.0 );
	assert( S_.size() == Sinit_.size() );
	S_.resize( S_.size() + recvMeshIndex.size(), temp );
	Sinit_.resize( Sinit_.size() + recvMeshIndex.size(), temp );
	j.setRecvPools( recvMeshIndex, selfDiffPoolIndex );
}

void StoichPools::addJunction( const Eref& e, const Qinfo* q, Id other )
{
	if ( !validateJunction( e.id(), other ) ) 
			return;

	StoichPools* otherSP = 
			reinterpret_cast< StoichPools* >( other.eref().data() );
	
	// Set up message
	innerConnectJunctions( e.id(), other, otherSP );

	// Work out reaction terms.
	//
	// This is an index into the rates_ array on the parent Stoich.
	vector< unsigned int > reacTerms;

	// This identifies the poolIndex for each reacTermIndex.
	// First of pair is reacTerm index, second is poolIndex.
	vector< pair< unsigned int, unsigned int > > reacPoolIndex;
	this->vBuildReacTerms( reacTerms, reacPoolIndex, other );
	junctions_.back().setReacTerms( reacTerms, reacPoolIndex );

	reacTerms.resize( 0 );
	reacPoolIndex.resize( 0 );
	otherSP->vBuildReacTerms( reacTerms, reacPoolIndex, e.id() );
	otherSP->junctions_.back().setReacTerms( reacTerms, reacPoolIndex );

	// Work out diffusion terms
	// Here the vectors are just the PoolIndices (as used within the Stoichs
	// to identify pools) for diffusing molecules. These map one-to-one
	// with each other and with data transfer terms.
	vector< unsigned int > selfDiffPoolIndex;
	vector< unsigned int > otherDiffPoolIndex;
	findDiffusionTerms( otherSP, selfDiffPoolIndex, otherDiffPoolIndex );
	junctions_.back().setDiffTerms( selfDiffPoolIndex );
	otherSP->junctions_.back().setDiffTerms( otherDiffPoolIndex );

	// Work out matching meshEntries. Virtual func. Voxelized solvers 
	// refer to their ChemMesh. Non-voxelized ones always have a single
	// meshEntry, and hold a dummy ChemMesh to do this calculation.
	// In these vectors the first entry of the pair is the index in the 
	// arriving // vector of rates modulo # of reacs, and the second is 
	// the target meshIndex.
	vector< unsigned int > selfMeshIndex; 
	vector< unsigned int > otherMeshIndex; 
	vector< VoxelJunction > selfMeshMap; 
	vector< VoxelJunction > otherMeshMap; 
	this->matchMeshEntries( otherSP, selfMeshIndex, selfMeshMap,
					otherMeshIndex, otherMeshMap );
	junctions_.back().setMeshIndex( selfMeshIndex, selfMeshMap );
	junctions_.back().setSendPools( selfMeshIndex, selfDiffPoolIndex );

	// Here we have to expand the S matrix to include these points.
	this->expandSforDiffusion( 
					otherMeshIndex, selfDiffPoolIndex, junctions_.back() );

	otherSP->junctions_.back().setMeshIndex( otherMeshIndex, otherMeshMap );
	otherSP->junctions_.back().setSendPools( otherMeshIndex, otherDiffPoolIndex );
	otherSP->junctions_.back().setRecvPools( otherMeshIndex, otherDiffPoolIndex );
	otherSP->expandSforDiffusion( 
			selfMeshIndex, otherDiffPoolIndex, otherSP->junctions_.back() );
}


void StoichPools::dropJunction( const Eref& e, const Qinfo* q, Id other )
{
	vDropJunction( e, q, other );
}

