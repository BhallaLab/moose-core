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
#include "SolverBase.h"
#include "../shell/Shell.h"
#include "../kinetics/lookupSizeFromMesh.h"

const Cinfo* SolverBase::initCinfo()
{
	static DestFinfo addJunction( "addJunction",
			"Add a junction between the current solver and the one whose"
			" Id is passed in.",
			new EpFunc1< SolverBase, Id >( &SolverBase::addJunction )
	);

	static DestFinfo dropJunction( "dropJunction",
			"Drops a junction between the current solver and the one whose"
			" Id is passed in. Ignores if no junction.",
			new EpFunc1< SolverBase, Id >( &SolverBase::dropJunction )
	);

	static DestFinfo reallocateSolver( "reallocateSolver",
		"Reallocates storage for solver. Needed when rebuilding. ",
		new EpFunc0< SolverBase >( &SolverBase::reallocateSolver )
	);

	static DestFinfo reconfigureJunctions( "reconfigureJunctions",
		"Goes through all junctions and updates their interfaces. "
		"Should be called whenever the reaction-diffusion system has "
		"been changed, for example, by remeshing or adding reactions.",
		new EpFunc0< SolverBase >( &SolverBase::reconfigureAllJunctions )
	);

	static FieldElementFinfo< SolverBase, SolverJunction > junction(
		"junction",
		"Handles how solvers communicate with each other in case of "
		"diffusion, motors, or reaction.",
		SolverJunction::initCinfo(),
		&SolverBase::getJunction,
		&SolverBase::setNumJunctions,
		&SolverBase::getNumJunctions,
		16
	);

	static Finfo* solverBaseFinfos[] = {
		&addJunction,		// DestFinfo
		&dropJunction,		// DestFinfo
		&reallocateSolver,	// DestFinfo
		&reconfigureJunctions,	// DestFinfo
		&junction			// FieldElement
	};

	static string doc[] = 
	{
			"Name", "SolverBase",
			"Author", "Upinder S. Bhalla, 2012, NCBS",
			"Description", "Pure virtual base class for handling "
			"solvers, with special emphasis toward reaction-diffusion "
			"systems spanning multiple solvers and processors. "
			"GslStoich is derived from this."
			"The main role of theis class is to set up the Junctions "
			"between solvers."
	};

	static Cinfo solverBaseCinfo(
		"SolverBase",
		Neutral::initCinfo(),
		solverBaseFinfos,
		sizeof( solverBaseFinfos ) / sizeof( Finfo* ),
		new ZeroSizeDinfo< int >(),
		doc, sizeof( doc ) / sizeof( string )
	);
	
	return &solverBaseCinfo;
}

static const Cinfo* solverBaseCinfo = SolverBase::initCinfo();

//////////////////////////////////////////////////////////////
// Class definitions
//////////////////////////////////////////////////////////////

SolverBase::SolverBase()
{;}

SolverBase::~SolverBase()
{;}

//////////////////////////////////////////////////////////////////////////
// Access functions for junctions
//////////////////////////////////////////////////////////////////////////
SolverJunction* SolverBase::getJunction( unsigned int i )
{
	static SolverJunction dummy;
	if ( i < junctions_.size() )
		return &junctions_[i];

	cout << "Warning: SolverBase::getJunction: Index: " << i << 
			" is out of range: " << junctions_.size() << endl;

	return &dummy;
}

void SolverBase::setNumJunctions( unsigned int v )
{
	cout << "Warning: SolverBase::setNumJunctions: Direct assignment "
		"of number not permitted, use addJunction instead.";
}	

unsigned int SolverBase::getNumJunctions() const
{
	return junctions_.size();
}

void SolverBase::handleJunctionPoolDelta( unsigned int fieldIndex,
		vector< double > v )
{
	vHandleJunctionPoolDelta( fieldIndex, v );
}

void SolverBase::handleJunctionPoolNum( unsigned int fieldIndex,
		vector< double > v )
{
	vHandleJunctionPoolNum( fieldIndex, v );
}

// me is self compartment, other is other compartment.
bool validateJunction( Id me, Id other )
{
	// Check that types match
	if ( !other.element()->cinfo()->isA( "SolverBase" ) ) {
		cout << "Warning: SolverBase::validateJunction: Other Id '" <<
				other.path() << " is not a StoichPool\n";
		return false;
	}
	// Check if junction already exists.
	Id myJunction( me.value() + 1);
	Id otherJunction( other.value() + 1);
	vector< Id > ret;
	myJunction.element()->getNeighbours( ret, junctionPoolDeltaFinfo() );
	if ( find( ret.begin(), ret.end(), otherJunction ) != ret.end() ) {
		cout << "Warning: SolverBase::validateJunction: junction " <<
		" already present from " << 
		me.path() << " to " << other.path() << endl;
		return false;
	}
	return true;
}

void SolverBase::findDiffusionTerms( 
				const SolverBase* otherSP,
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

void SolverBase::innerConnectJunctions( 
			Id me, Id other, SolverBase* otherSP )
{
	Shell* shell = reinterpret_cast< Shell* >( Id().eref().data() );
	Id myJunction( me.value() + 1);
	Id otherJunction( other.value() + 1);
	// Set up message
	ObjId myOid( myJunction, getNumJunctions() );
	ObjId otherOid( otherJunction, otherSP->getNumJunctions() );
	MsgId mid = shell->doAddMsg( "single", myOid, "masterJunction", 
					otherOid, "followerJunction" );
	assert( mid != Msg::bad );

	// Make junction entries
	junctions_.resize( junctions_.size() + 1 );
	Id myCompt = getCompt( me );
	Id otherCompt = getCompt( other );
	junctions_.back().setCompartments( myCompt, otherCompt );
	otherSP->junctions_.resize( otherSP->junctions_.size() + 1 );
	otherSP->junctions_.back().setCompartments( otherCompt, myCompt );
}

// This is called only on the master solver, the one which does the
// diffusion calculations. To do so, this solver expands its own pool
// matrices S and Sinit to include the abutting voxels.
// The follower solver sends messages to put pool #s into the abutting
// voxels, and gets back changes in these pool #s. Does not do the
// diffusion calculations and does not expand.
void SolverBase::addJunction( const Eref& e, const Qinfo* q, Id otherSolver)
{
	if ( !validateJunction( e.id(), otherSolver ) ) 
			return;

	SolverBase* otherSP = 
			reinterpret_cast< SolverBase* >( otherSolver.eref().data() );
	
	// Set up message
	innerConnectJunctions( e.id(), otherSolver, otherSP );

	configureJunction( e.id(), otherSolver,
					junctions_.back(), otherSP->junctions_.back() );
}


// Should only be called on the master junction.
void SolverBase::configureJunction( Id selfSolver, Id otherSolver, 
		SolverJunction& junc, SolverJunction& otherJunc )
{
	SolverBase* otherSP = 
		reinterpret_cast< SolverBase* >( otherSolver.eref().data() );
	
	// Work out reaction terms.
	vector< Id > pools;
	this->findPoolsOnOther( otherSolver, pools );
	otherSP->setLocalCrossReactingPools( pools );

	pools.clear();
	otherSP->findPoolsOnOther( selfSolver, pools );
	this->setLocalCrossReactingPools( pools );

	// Work out diffusion terms
	// Here the vectors are just the PoolIndices (as used within the Stoichs
	// to identify pools) for diffusing molecules. These map one-to-one
	// with each other and with data transfer terms.
	vector< unsigned int > selfDiffPoolIndex;
	vector< unsigned int > otherDiffPoolIndex;
	findDiffusionTerms( otherSP, selfDiffPoolIndex, otherDiffPoolIndex );
	junc.setDiffTerms( selfDiffPoolIndex );
	otherJunc.setDiffTerms( otherDiffPoolIndex );

	// Work out matching meshEntries. Virtual func. Voxelized solvers 
	// refer to their ChemCompt. Non-voxelized ones always have a single
	// meshEntry, and hold a dummy ChemCompt to do this calculation.
	// In these vectors the first entry of the pair is the index in the 
	// arriving // vector of rates modulo # of reacs, and the second is 
	// the target meshIndex.
	vector< unsigned int > selfMeshIndex; 
	vector< unsigned int > otherMeshIndex; 
	vector< VoxelJunction > selfMeshMap; 
	vector< VoxelJunction > otherMeshMap; 
	this->matchMeshEntries( otherSP, selfMeshIndex, selfMeshMap,
					otherMeshIndex, otherMeshMap );
	junc.setMeshMap( selfMeshMap );
	junc.setSendPools( selfMeshIndex, selfDiffPoolIndex );
	// Here we have to expand the S matrix to include these points.
	// This function also sets the abutRecvIndex vector.
	this->expandSforDiffusion( 
					otherMeshIndex, selfDiffPoolIndex, junc );

	otherJunc.setMeshMap( otherMeshMap );
	otherJunc.setSendPools( otherMeshIndex, otherDiffPoolIndex );
	// The junction on the otherSP does not expand the S matrix.
	
	
	// Here we tell the affected derived classes to update anything that
	// depends on the Junctions, such as the GslStoich::ode_ array.
	this->updateJunctionInterface( selfSolver.eref() );
	otherSP->updateJunctionInterface( otherSolver.eref() );
}


void SolverBase::dropJunction( const Eref& e, const Qinfo* q, Id other )
{
	vDropJunction( e, q, other );
}

/*
// Utility function: return the compartment in which the specified
// object is located.
// Simply traverses the tree toward the root till it finds a
// compartment. Pools use a special msg, but this works for reacs too.
Id getCompt( Id id )
{
	const Element* e = id.element();
	if ( e->cinfo()->isA( "PoolBase" ) ) {
		vector< Id > neighbours;
		if ( e->getNeighbours( neighbours, e->cinfo()->findFinfo( "requestSize" ) ) == 1 ) {
			Id pa = Neutral::parent( neighbours[0].eref() ).id;
			if ( pa.element()->cinfo()->isA( "ChemCompt" ) )
				return pa;
		}
	}
	Id pa = Neutral::parent( id.eref() ).id;
	if ( pa == Id() )
		return pa;
	else if ( pa.element()->cinfo()->isA( "ChemCompt" ) )
		return pa;
	return getCompt( pa );
}
*/

/**
 * Scans through all junctions and if they are master junctions, 
 * reconfigures them. This is used whenever any part of the reac-diff
 * system has been altered.
 */
void SolverBase::reconfigureAllJunctions( const Eref& e, const Qinfo* q )
{
	Id selfSolver = e.id();
	Id myJunction( selfSolver.value() + 1);
	vector< Id > ret;
	
	// Reallocate the pool vector from scratch.
	// Go through msg list
	// Get src and dest ObjId using the srcToDestPairs function.
	// Identify the junction indices from this.
	// Call configureJunction using this info.
	
	const vector < MsgFuncBinding >* mb = 
			myJunction.element()->getMsgAndFunc( 
							junctionPoolDeltaFinfo()->getBindIndex() );
	assert( mb != 0 );
	for ( vector< MsgFuncBinding >::const_iterator 
					i = mb->begin(); i != mb->end(); ++i ) {
		const Msg* m = Msg::getMsg( i->mid );
		Element* selfE = m->e1();
		Element* otherE = m->e2();
		assert( selfE == myJunction.element() );
		vector< DataId > src;
		vector< DataId > dest;
		unsigned int num = m->srcToDestPairs( src, dest );
		assert( num == 1 );
		SolverJunction* selfJ = reinterpret_cast< SolverJunction * >(
				Eref( selfE, src[0] ).data() );
		SolverJunction* otherJ = reinterpret_cast< SolverJunction * >(
				Eref( otherE, dest[0] ).data() );
		assert( selfJ->getMyCompartment() == otherJ->getOtherCompartment());
		assert( selfJ->getOtherCompartment() == otherJ->getMyCompartment());
		assert( selfJ->getMyCompartment() == getCompt( selfSolver ) );
		Id otherSolver( otherE->id().value() - 1 );
		configureJunction( selfSolver, otherSolver, *selfJ, *otherJ );
	}
}

void SolverBase::reallocateSolver( const Eref& e, const Qinfo* q )
{
	this->innerReallocateSolver( e );
}
