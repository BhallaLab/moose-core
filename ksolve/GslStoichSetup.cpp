/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "StoichHeaders.h"
#include <gsl/gsl_errno.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_odeiv.h>
#include "SolverBase.h"
#include "VoxelPools.h"
#include "OdeSystem.h"
#include "../shell/Shell.h"
#include "../shell/Wildcard.h"
#include "../mesh/MeshEntry.h"
#include "../mesh/Boundary.h"
#include "../mesh/ChemCompt.h"
#include "GslStoich.h"


///////////////////////////////////////////////////
// Field function definitions
///////////////////////////////////////////////////

///// Here gen all odes, up to 3 at a time, sorted by ascending Id.
unsigned int GslStoich::generateOdes()
{
	vector< Id > compts = coreStoich_.getOffSolverCompts();
	sort( compts.begin(), compts.end() ); // Just to be sure.
	ode_.clear();
	// Zero order
	vector< Id > comptSignature( 0 );
	ode_.push_back( OdeSystem( &coreStoich_, comptSignature ) );

	// First order
	comptSignature.resize( 1 );
	for ( unsigned int i = 0; i < compts.size(); ++i ) {
		comptSignature[0] = compts[i];
		ode_.push_back( OdeSystem( &coreStoich_, comptSignature ) );
	}

	// Second order. Note that the sig has to be in ascending order of Id.
	// The compts are already in ascending order.
	comptSignature.resize( 2 );
	for ( unsigned int i = 1; i < compts.size(); ++i ) {
		comptSignature[0] = compts[i - 1];
		for ( unsigned int j = i; j < compts.size(); ++j ) {
			comptSignature[1] = compts[j];
			ode_.push_back( OdeSystem( &coreStoich_, comptSignature ) );
		}
	}

	// Third order. Again, keep the sig in ascending order.
	comptSignature.resize( 3 );
	for ( unsigned int i = 2; i < compts.size(); ++i ) {
		comptSignature[0] = compts[ i - 2 ];
		for ( unsigned int j = i; j < compts.size(); ++j ) {
			comptSignature[1] = compts[j - 1];
			for ( unsigned int k = j; k < compts.size(); ++k ) {
				comptSignature[2] = compts[k];
				ode_.push_back( OdeSystem( &coreStoich_, comptSignature ) );
			}
		}
	}
	return ode_.size();
}

void GslStoich::setElist( const Eref& e, const Qinfo* q, vector< Id > elist)
{
	if ( isInitialized_ )
		return;
	path_ = "elist";
	ode_.clear();
	pools_.clear();
	y_.clear();
	isInitialized_ = 0;
	compartmentId_ = getCompt( e.id() );
	if ( compartmentId_ != Id() ) {
		diffusionMesh_ = reinterpret_cast< ChemCompt* >( 
						compartmentId_.eref().data() );
		// localMeshEntries is really supposed to map the global indexing
		// of meshEntries onto the local indices used in this object.
		localMeshEntries_.resize( diffusionMesh_->getNumEntries(), 0 );
	} else {
		localMeshEntries_.resize( 1, 0 );
	}
	// Within the setPath function, the stoich calls the allocatePools 
	// function below
	coreStoich_.setElist( e, this, elist );

	// Set up the vector of pools that diffuse.
	diffusingPoolIndices_.clear();
	for ( unsigned int i = 0; i < coreStoich_.getNumVarPools(); ++i ) {
		if ( coreStoich_.getDiffConst( i ) > 0 ) {
			diffusingPoolIndices_.push_back( i );
		}
	}

	generateOdes();
	assert( ode_.size() > 0 );

	unsigned int numPools = coreStoich()->getNumVarPools() + 
			coreStoich()->getNumProxyPools();
	for ( unsigned int i = 0; i < ode_.size(); ++i ) {
		ode_[i].setMethod( method_ );
		ode_[i].reinit( this,
			&GslStoich::gslFunc,
			numPools, absAccuracy_, relAccuracy_ );
	}
	isInitialized_ = ( pools_.size() > 0 );
}

void GslStoich::setPath( const Eref& e, const Qinfo* q, string path )
{
	if ( isInitialized_ && path_ == path )
		return;
	vector< Id > elist;
	wildcardFind( path, elist );
	setElist( e, q, elist );
	path_ = path;
}

void GslStoich::allocatePools( unsigned int numPools )
{
	pools_.resize( localMeshEntries_.size() );
	y_.resize( localMeshEntries_.size() );
	// unsigned int numPools = coreStoich()->getNumAllPools() + coreStoich()->getNumProxyPools();
	for ( unsigned int i = 0; i < localMeshEntries_.size(); ++i ) {
		// const OdeSystem& ode = ode_[ pools_[i].getSolver() ];
		pools_[i].resizeArrays( numPools );
		y_[i].resize( numPools, 0 );
	}
}

///////////////////////////////////////////////////
// Dest function definitions
///////////////////////////////////////////////////
/**
 * This function should also set up the sizes, and it should be at 
 * allocate, not reinit time.
 */
/*
void GslStoich::initialize( const Eref& e, const Qinfo* q )
{
	
	stoich_ = reinterpret_cast< StoichCore* >( stoichId.eref().data() );

	unsigned int nVarPools = stoich_->getNumVarPools();
	// stoich_->clearFlux();
	resizeArrays( stoich_->getNumAllPools() + 
					stoich_->getOffSolverPools().size() );
	vector< double > temp( stoich_->getNumVarPools(), 0.0 );
	y_.resize( numMeshEntries(), temp );

	isInitialized_ = 1;
        // Allocate GSL functions if not already allocated,
        // otherwise reset the reusable ones
        assert( gslStepType_ != 0 );
        if ( gslStep_ )
        {
            gsl_odeiv_step_free(gslStep_);
        }
        
        gslStep_ = gsl_odeiv_step_alloc( gslStepType_, nVarPools );
        
   	assert( gslStep_ != 0 );
        if ( !gslEvolve_ )
        {
            gslEvolve_ = gsl_odeiv_evolve_alloc(nVarPools);
        }
        else
        {
            gsl_odeiv_evolve_reset(gslEvolve_);
        }
        assert(gslEvolve_ != 0);
        
        if ( !gslControl_ )
        {
            gslControl_ = gsl_odeiv_control_y_new( absAccuracy_, relAccuracy_ );
        }
        else 
        {
            gsl_odeiv_control_init(gslControl_,absAccuracy_, relAccuracy_, 1, 0);
        }
        assert(gslControl_!= 0);
        
        
	gslSys_.function = &GslStoich::gslFunc;
	gslSys_.jacobian = 0;
	gslSys_.dimension = nVarPools;
	gslSys_.params = static_cast< void* >( this );
	// gslSys_.params = static_cast< void* >( s );
}
*/


///////////////////////////////////////////////////////////////////////////
// Junction operations
///////////////////////////////////////////////////////////////////////////

/**
 * Utility function called by the manager. Sets up the mutual messaging,
 * and then configures the junctions on either end.
 * These functions should really migrate down to the parent class.
 */
void GslStoich::vAddJunction( const Eref& e, const Qinfo* q, Id other )
{
}

void GslStoich::vDropJunction( const Eref& e, const Qinfo* q, Id other )
{
}

unsigned int GslStoich::selectOde( const vector< Id >& sig ) const
{
	if ( sig.size() == 0 ) return 0;
	for ( unsigned int i = 1; i < ode_.size(); ++i ) {
		if ( ode_[i].compartmentSignature_ == sig )
			return i;
	}
	assert( 0 ); // Should always find a match.
	return 0;
}

void addToPoolSig( vector< vector< Id > >& sig, 
				Id compt, 
				const vector< unsigned int >& voxels )
{
	for ( vector< unsigned int >::const_iterator
					i = voxels.begin(); i != voxels.end(); ++i ) {
		sig[*i].push_back( compt );
	}
}

/**
 * Virtual function to do any local updates to the stoich following changes
 * to the Junctions.
 * Here key role is to check for cross-solver reactions handled by local
 * stoichs. It uses this to assign the correct ode_ entry.
 */
void GslStoich::updateJunctionInterface( const Eref& e )
{
	for ( vector< VoxelPools >::iterator 
			i = pools_.begin(); i != pools_.end(); ++i )
		i->setSolver( 0 ); // Default all to the solver without any X-reacs
	// Now march through all junctions to figure out which combination of
	// compartments applies to which pools.
	
	// sig is the signature of each voxel. Starts out empty, and 
	// compartments are pushed into it by checking with each junction, 
	// since a given voxel may have cross-reactions with multiple other
	// compartments.
	vector< vector< Id > > sig( localMeshEntries_.size() );
	for ( unsigned int i =0; i < getNumJunctions(); ++i ) {
		SolverJunction* sj = getJunction( i );
		if ( sj->remoteReacPools().size() > 0 ) {
			addToPoolSig( sig, sj->getOtherCompartment(), 
							sj->sendMeshIndex() );
		}
	}
	// Clean up using typically grotesque C++ STL syntax.
	for ( vector< vector< Id > >::iterator 
					i = sig.begin(); i != sig.end(); ++i ) {
		sort( i->begin(), i->end() );
		i->erase( unique( i->begin(), i->end() ), i->end() );
	}

 	// Go through all pools and assign ode_ entry.
	assert( y_.size() >= sig.size() );
	assert( pools_.size() >= sig.size() );
	for ( unsigned int i = 0; i < sig.size(); ++i ) {
		unsigned int solver = selectOde( sig[i] );
		pools_[i].setSolver( solver );
		unsigned int np = 
						ode_[solver].stoich_->getNumAllPools() + 
						ode_[solver].stoich_->getNumProxyPools();
		pools_[i].resizeArrays( np );
		y_[i].resize( np );
	}
	junctionsNotReady_ = true;
}

// Return elist of pools on Other solver that are reactants on this solver.
void GslStoich::findPoolsOnOther( Id other, vector< Id >& pools ) 
{
	unsigned int nj = getNumJunctions();
	assert( nj > 0 );
	pools.clear();
	vector< unsigned int > poolIndex;
	Id otherCompt = getCompt( other );
	assert( otherCompt != Id() );
	const StoichCore* stoich = coreStoich();
	const vector< Id >& op = stoich->getOffSolverPools();
	for ( vector< Id >::const_iterator i = op.begin(); i != op.end(); ++i )
	{
		if ( getCompt( *i ) == otherCompt ) {
			pools.push_back( *i );
			poolIndex.push_back( stoich->convertIdToPoolIndex( *i ) );
		}
	}
	getJunction( nj - 1 )->setRemoteReacPools( poolIndex );
}

const StoichCore* GslStoich::coreStoich() const
{
	return &coreStoich_;
}

void GslStoich::setLocalCrossReactingPools( const vector< Id >& pools )
{
	unsigned int nj = getNumJunctions();
	assert( nj > 0 );
	// convert pools to local pool indices
	
	const StoichCore* stoich = coreStoich();
	vector< unsigned int > poolIndex;
	for ( vector< Id >::const_iterator 
					i = pools.begin(); i != pools.end(); ++i )
		poolIndex.push_back( stoich->convertIdToPoolIndex( *i ) );

	getJunction( nj - 1 )->setLocalReacPools( poolIndex );
}

void GslStoich::vBuildDiffTerms( map< string, unsigned int >& diffTerms ) 
		const
{
	// Shouldn't have to redo, this call should happen once
	for ( vector< OdeSystem >::const_iterator
					i = ode_.begin(); i != ode_.end(); ++i )
		i->stoich_->buildDiffTerms( diffTerms );
}

// Virtual func figures out which meshEntries line up, passes the job to
// the relevant ChemCompt.
void GslStoich::matchMeshEntries( 
	SolverBase* other,
	vector< unsigned int >& selfMeshIndex, 
	vector< VoxelJunction >& selfMeshMap, 
	vector< unsigned int >& otherMeshIndex, 
	vector< VoxelJunction >& otherMeshMap
	) const
{
	// This vector is a map of meshIndices in this to other compartment.
	vector< VoxelJunction > meshMatch;
	assert( compartmentMesh() );
	assert( other->compartmentMesh() );
	diffusionMesh_->buildJunction( other->compartmentMesh(), meshMatch );
	// First, extract the meshIndices. Need to make sure they are unique.
	for ( vector< VoxelJunction>::iterator i = meshMatch.begin(); 
					i != meshMatch.end(); ++i ){
		selfMeshIndex.push_back( i->first );
		otherMeshIndex.push_back( i->second );
	}
	sort( selfMeshIndex.begin(), selfMeshIndex.end() );
	vector< unsigned int >::iterator end = 
			unique( selfMeshIndex.begin(), selfMeshIndex.end() );
	selfMeshIndex.resize( end - selfMeshIndex.begin() );
	map< unsigned int, unsigned int > selfMeshLookup;
	for ( unsigned int i = 0; i < selfMeshIndex.size(); ++i )
		selfMeshLookup[ selfMeshIndex[i] ] = i;

	sort( otherMeshIndex.begin(), otherMeshIndex.end() );
	end = unique( otherMeshIndex.begin(), otherMeshIndex.end() );
	otherMeshIndex.resize( end - otherMeshIndex.begin() );
	map< unsigned int, unsigned int > otherMeshLookup;
	for ( unsigned int i = 0; i < otherMeshIndex.size(); ++i )
		otherMeshLookup[ otherMeshIndex[i] ] = i;

	// Now stuff values into the meshMaps.
	for ( vector< VoxelJunction>::iterator i = meshMatch.begin(); 
					i != meshMatch.end(); ++i ){
		map< unsigned int, unsigned int >::iterator k = 
				otherMeshLookup.find( i->second );
		assert( k != otherMeshLookup.end() );
		selfMeshMap.push_back( 
					VoxelJunction( k->second, i->first, i->diffScale ) );

		k =	selfMeshLookup.find( i->first );
		assert( k != selfMeshLookup.end() );
		otherMeshMap.push_back( 
					VoxelJunction( k->second, i->second, i->diffScale ) );
	}

	// Now update the S_ and Sinit_ matrices to deal with the added 
	// meshEntries.
	
}

ChemCompt* GslStoich::compartmentMesh() const
{
	return diffusionMesh_;
}
///////////////////////////////////////////////////
// Remesh
///////////////////////////////////////////////////
//
// This function has to be updated to deal with changed boundaries. Ugh.
void GslStoich::meshSplit( 
				vector< double > initConcs,  // in milliMolar
				vector< double > vols,		// in m^3
				vector< unsigned int > localEntryList )
{
	assert ( coreStoich()->getNumAllPools() == initConcs.size() );
	assert( vols.size() == localEntryList.size() );
	unsigned int numLocalVoxels = vols.size();

	pools_.resize( numLocalVoxels );
	y_.resize( numLocalVoxels );
	for ( unsigned int i = 0; i < numLocalVoxels; ++i ) {
		// Need to put in a check to see if there is an interface to another
		// compartment on these voxels
		//ode_[ pools_[i].getSolver() ].stoich_->getOffSolverPools().size();
		// I need somehow to assign the correct solver for each voxel.
		const OdeSystem& ode = ode_[ pools_[i].getSolver() ];
		unsigned int numPools = ode.stoich_->getNumAllPools() + 
				ode.stoich_->getNumProxyPools();
		pools_[i].resizeArrays( numPools );
		y_[i].resize( numPools );
		double v = vols[i] * NA;
		double* sInit = pools_[i].varSinit();
		for ( unsigned int j = 0; j < initConcs.size(); ++j ) 
			sInit[j] = initConcs[j] * v;
		pools_[i].reinit();
		pools_[i].setSolver( 0 );
	}
	localMeshEntries_ = localEntryList;
}


void GslStoich::remesh( const Eref& e, const Qinfo* q,
	double oldVol,
	unsigned int numTotalEntries, unsigned int startEntry, 
	vector< unsigned int > localIndices, vector< double > vols )
{
	if ( e.index().value() != 0 || !isInitialized_ ) {
		return;
	}
	// cout << "GslStoich::remesh for " << e << endl;
	assert( vols.size() > 0 );
	// Here we could have a change in the meshing, or in the volumes.
	// Either way we need to do scaling.
	unsigned int numPools = coreStoich()->getNumAllPools();
	vector< double > initConcs( numPools, 0.0 );
	vector< unsigned int > localEntryList( vols.size(), 0 );
	for ( unsigned int i = 0; i < vols.size(); ++i )
			localEntryList[i] = i;

	for ( unsigned int i = 0; i < numPools; ++i ) {
		initConcs[i] = pools_[0].Sinit()[i] / ( NA * oldVol );
	}
	meshSplit( initConcs, vols, localEntryList );
	vector< double > temp( numPools, 0.0 );
	y_.resize( vols.size(), temp );

	coreStoich_.updateRatesAfterRemesh();
	junctionsNotReady_ = true;
}

// Inherited virtual function.
void GslStoich::expandSforDiffusion(
	const vector< unsigned int > & otherMeshIndex,
	const vector< unsigned int > & selfDiffPoolIndex,
	SolverJunction& j )
{
	if ( !diffusionMesh_ )
		return;
	unsigned int numVoxels = diffusionMesh_->getNumEntries();
	unsigned int numCorePoolEntries = coreStoich()->getNumAllPools();
	unsigned int numPoolStructs = pools_.size();
	vector< unsigned int > abutMeshIndex( otherMeshIndex.size(), 0 );
	for ( unsigned int i = 0; i < otherMeshIndex.size(); ++i )
		abutMeshIndex[i] = i + numPoolStructs;
	if ( selfDiffPoolIndex.size() > 0 )
		numPoolStructs += abutMeshIndex.size();
	pools_.resize( numPoolStructs );
	y_.resize( numPoolStructs );
	for ( unsigned int i = numVoxels; i < pools_.size(); ++i ) {
		pools_[i].resizeArrays( numCorePoolEntries );
		y_[i].resize( numCorePoolEntries );
	}
	j.setAbutPools( abutMeshIndex, selfDiffPoolIndex );
}

/**
 * This function reallocates the storage for the solver. It is typically
 * called after a series of remeshing calls through the entire reaction
 * system, which require it to flush the old allocations and rebuild.
 */
void GslStoich::innerReallocateSolver( const Eref& e )
{
	unsigned int numVoxels = 1;
	if ( diffusionMesh_ ) {
		numVoxels = diffusionMesh_->getNumEntries();
		diffusionMesh_->clearExtendedMeshEntrySize();
	}

	assert( numVoxels > 0 );
	vector< double > vols( numVoxels );
	for ( unsigned int i = 0; i < numVoxels; ++i ) {
		vols[i] = diffusionMesh_->getMeshEntrySize( i );
	}
	double oldVol = vols[0];
	remesh( e, 0, oldVol, 0, 0, localMeshEntries_, vols );
}
