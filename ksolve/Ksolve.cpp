/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/
#include "header.h"
#ifdef USE_GSL
#include <gsl/gsl_errno.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_odeiv2.h>
#endif

#include "OdeSystem.h"
#include "VoxelPoolsBase.h"
#include "VoxelPools.h"
#include "ZombiePoolInterface.h"

#include "RateTerm.h"
#include "FuncTerm.h"
#include "SparseMatrix.h"
#include "KinSparseMatrix.h"
#include "Stoich.h"

#include "Ksolve.h"

const unsigned int OFFNODE = ~0;

const Cinfo* Ksolve::initCinfo()
{
		///////////////////////////////////////////////////////
		// Field definitions
		///////////////////////////////////////////////////////
		
		static ValueFinfo< Ksolve, Id > stoich (
			"stoich",
			"Stoichiometry object for handling this reaction system.",
			&Ksolve::setStoich,
			&Ksolve::getStoich
		);

		static ValueFinfo< Ksolve, Id > dsolve (
			"dsolve",
			"Diffusion solver object handling this reactin system.",
			&Ksolve::setDsolve,
			&Ksolve::getDsolve
		);

		static ReadOnlyValueFinfo< Ksolve, unsigned int > numLocalVoxels(
			"numLocalVoxels",
			"Number of voxels in the core reac-diff system, on the "
			"current solver. ",
			&Ksolve::getNumLocalVoxels
		);
		static LookupValueFinfo< 
				Ksolve, unsigned int, vector< double > > nVec(
			"nVec",
			"vector of pool counts. Index specifies which voxel.",
			&Ksolve::setNvec,
			&Ksolve::getNvec
		);
		static ValueFinfo< Ksolve, unsigned int > numAllVoxels(
			"numAllVoxels",
			"Number of voxels in the entire reac-diff system, "
			"including proxy voxels to represent abutting compartments.",
			&Ksolve::setNumAllVoxels,
			&Ksolve::getNumAllVoxels
		);

		static ValueFinfo< Ksolve, unsigned int > numPools(
			"numPools",
			"Number of molecular pools in the entire reac-diff system, "
			"including variable, function and buffered.",
			&Ksolve::setNumPools,
			&Ksolve::getNumPools
		);

		///////////////////////////////////////////////////////
		// DestFinfo definitions
		///////////////////////////////////////////////////////

		static DestFinfo process( "process",
			"Handles process call",
			new ProcOpFunc< Ksolve >( &Ksolve::process ) );
		static DestFinfo reinit( "reinit",
			"Handles reinit call",
			new ProcOpFunc< Ksolve >( &Ksolve::reinit ) );
		
		///////////////////////////////////////////////////////
		// Shared definitions
		///////////////////////////////////////////////////////
		static Finfo* procShared[] = {
			&process, &reinit
		};
		static SharedFinfo proc( "proc",
			"Shared message for process and reinit",
			procShared, sizeof( procShared ) / sizeof( const Finfo* )
		);

	static Finfo* ksolveFinfos[] =
	{
		&stoich,			// Value
		&dsolve,			// Value
		&numLocalVoxels,	// ReadOnlyValue
		&nVec,				// LookupValue
		&numAllVoxels,		// ReadOnlyValue
		&numPools,			// Value
		&proc,				// SharedFinfo
	};
	
	static Dinfo< Ksolve > dinfo;
	static  Cinfo ksolveCinfo(
		"Ksolve",
		Neutral::initCinfo(),
		ksolveFinfos,
		sizeof(ksolveFinfos)/sizeof(Finfo *),
		&dinfo
	);

	return &ksolveCinfo;
}

static const Cinfo* ksolveCinfo = Ksolve::initCinfo();

//////////////////////////////////////////////////////////////
// Class definitions
//////////////////////////////////////////////////////////////

Ksolve::Ksolve()
	: 
		pools_( 1 ),
		startVoxel_( 0 ),
		stoich_(),
		stoichPtr_( 0 ),
		dsolve_(),
		dsolvePtr_( 0 )
{;}

Ksolve::~Ksolve()
{;}

//////////////////////////////////////////////////////////////
// Field Access functions
//////////////////////////////////////////////////////////////

Id Ksolve::getStoich() const
{
	return stoich_;
}

void Ksolve::setStoich( Id stoich )
{
	assert( stoich.element()->cinfo()->isA( "Stoich" ) );
	stoich_ = stoich;
	stoichPtr_ = reinterpret_cast< const Stoich* >( stoich.eref().data() );
}

Id Ksolve::getDsolve() const
{
	return dsolve_;
}

void Ksolve::setDsolve( Id dsolve )
{
	assert( dsolve.element()->cinfo()->isA( "Dsolve" ) );
	dsolve_ = dsolve;
	dsolvePtr_ = 
		reinterpret_cast< ZombiePoolInterface* >( dsolve.eref().data() );
}

unsigned int Ksolve::getNumLocalVoxels() const
{
	return pools_.size();
}

unsigned int Ksolve::getNumAllVoxels() const
{
	return pools_.size(); // Need to redo.
}

// If we're going to do this, should be done before the zombification.
void Ksolve::setNumAllVoxels( unsigned int numVoxels )
{
	if ( numVoxels == 0 ) {
		return;
	}
	pools_.resize( numVoxels );
}

vector< double > Ksolve::getNvec( unsigned int voxel) const
{
	static vector< double > dummy;
	if ( voxel < pools_.size() ) {
		return const_cast< VoxelPools* >( &( pools_[ voxel ] ) )->Svec();
	}
	return dummy;
}

void Ksolve::setNvec( unsigned int voxel, vector< double > nVec )
{
	if ( voxel < pools_.size() ) {
		if ( nVec.size() != pools_[voxel].size() ) {
			cout << "Warning: Ksolve::setNvec: size mismatch ( " <<
				nVec.size() << ", " << pools_[voxel].size() << ")\n";
			return;
		}
		double* s = pools_[voxel].varS();
		for ( unsigned int i = 0; i < nVec.size(); ++i )
			s[i] = nVec[i];
	}
}
/*
void Ksolve::setNumAllVoxels( unsigned int numVoxels )
{
	if ( numVoxels == 0 ) {
		return;
	}
	// Preserve the number of pool species.
	unsigned int numPoolSpecies = pools_[0].size();
	// Preserve the concInit.
	vector< double > nInit( numPoolSpecies );
	for ( unsigned int i = 0; i < numPoolSpecies; ++i ) {
		nInit[i] = pools_[0].Sinit()[i]; 
	}
	
	// Later do the node allocations.
	pools_.clear();
	pools_.resize( numVoxels );
	if ( !stoichPtr_ )
		return;
	// assert( stoichPtr_ );
	OdeSystem ode;
#ifdef USE_GSL
	ode.gslSys.function = &VoxelPools::gslFunc;
   	ode.gslSys.jacobian = 0;
	ode.gslSys.dimension = stoichPtr_->getNumVarPools();
	// This cast is needed because the C interface for GSL doesn't 
	// use const void here.
   	ode.gslSys.params = const_cast< Stoich* >( stoichPtr_ );
	if ( ode.method == "rk5" ) {
		ode.gslStep = gsl_odeiv2_step_rkf45;
	}
#endif
	for ( unsigned int i = 0 ; i < numVoxels; ++i ) {
		pools_[i].resizeArrays( numPoolSpecies );
		pools_[i].setStoich( stoichPtr_, &ode );
		for ( unsigned int j = 0; j < numPoolSpecies; ++j ) {
			pools_[i].varSinit()[j] = nInit[j];
		}
	}
}
*/

//////////////////////////////////////////////////////////////
// Process operations.
//////////////////////////////////////////////////////////////
void Ksolve::process( const Eref& e, ProcPtr p )
{
	if ( dsolvePtr_ ) {
		vector< double > dvalues( 4 );
		vector< double > kvalues( 4 );
		kvalues[0] = dvalues[0] = 0;
		kvalues[1] = dvalues[1] = getNumLocalVoxels();
		kvalues[2] = dvalues[2] = 0;
		kvalues[3] = dvalues[3] = stoichPtr_->getNumAllPools();
		dsolvePtr_->getBlock( dvalues );
		/*
		getBlock( kvalues );
		vector< double >::iterator d = dvalues.begin() + 4;
		for ( vector< double >::iterator 
				k = kvalues.begin() + 4; k != kvalues.end(); ++k )
				*k++ = ( *k + *d )/2.0
		setBlock( kvalues );
		*/
		setBlock( dvalues );
		for ( vector< VoxelPools >::iterator 
					i = pools_.begin(); i != pools_.end(); ++i ) {
			i->advance( p );
		}
		getBlock( kvalues );
	
		dsolvePtr_->setBlock( kvalues );
	} else {

		for ( vector< VoxelPools >::iterator 
					i = pools_.begin(); i != pools_.end(); ++i ) {
			i->advance( p );
		}
	}
}

void Ksolve::reinit( const Eref& e, ProcPtr p )
{
	for ( vector< VoxelPools >::iterator 
					i = pools_.begin(); i != pools_.end(); ++i ) {
		i->reinit();
	}
}
//////////////////////////////////////////////////////////////
// Solver ops
//////////////////////////////////////////////////////////////

unsigned int Ksolve::getPoolIndex( const Eref& e ) const
{
	return stoichPtr_->convertIdToPoolIndex( e.id() );
}

unsigned int Ksolve::getVoxelIndex( const Eref& e ) const
{
	unsigned int ret = e.dataIndex();
	if ( ret < startVoxel_  || ret >= startVoxel_ + pools_.size() ) 
		return OFFNODE;
	return ret - startVoxel_;
}

//////////////////////////////////////////////////////////////
// Zombie Pool Access functions
//////////////////////////////////////////////////////////////

void Ksolve::setN( const Eref& e, double v )
{
	unsigned int vox = getVoxelIndex( e );
	if ( vox != OFFNODE )
		pools_[vox].setN( getPoolIndex( e ), v );
}

double Ksolve::getN( const Eref& e ) const
{
	unsigned int vox = getVoxelIndex( e );
	if ( vox != OFFNODE )
		return pools_[vox].getN( getPoolIndex( e ) );
	return 0.0;
}

void Ksolve::setNinit( const Eref& e, double v )
{
	unsigned int vox = getVoxelIndex( e );
	if ( vox != OFFNODE )
		pools_[vox].setNinit( getPoolIndex( e ), v );
}

double Ksolve::getNinit( const Eref& e ) const
{
	unsigned int vox = getVoxelIndex( e );
	if ( vox != OFFNODE )
		return pools_[vox].getNinit( getPoolIndex( e ) );
	return 0.0;
}

void Ksolve::setDiffConst( const Eref& e, double v )
{
		; // Do nothing.
}

double Ksolve::getDiffConst( const Eref& e ) const
{
		return 0;
}

void Ksolve::setNumPools( unsigned int numPoolSpecies )
{
	assert( stoichPtr_ );
	OdeSystem ode;
#ifdef USE_GSL
	ode.gslSys.function = &VoxelPools::gslFunc;
   	ode.gslSys.jacobian = 0;
	ode.gslSys.dimension = stoichPtr_->getNumAllPools();
	// This cast is needed because the C interface for GSL doesn't 
	// use const void here.
   	ode.gslSys.params = const_cast< Stoich* >( stoichPtr_ );
	if ( ode.method == "rk5" ) {
		ode.gslStep = gsl_odeiv2_step_rkf45;
	}
#endif
	unsigned int numVoxels = pools_.size();
	for ( unsigned int i = 0 ; i < numVoxels; ++i ) {
		pools_[i].resizeArrays( numPoolSpecies );
		pools_[i].setStoich( stoichPtr_, &ode );
	}
}

unsigned int Ksolve::getNumPools() const
{
	if ( pools_.size() > 0 )
		return pools_[0].size();
	return 0;
}

void Ksolve::getBlock( vector< double >& values ) const
{
	unsigned int startVoxel = values[0];
	unsigned int numVoxels = values[1];
	unsigned int startPool = values[2];
	unsigned int numPools = values[3];

	assert( startVoxel >= startVoxel_ );
	assert( numVoxels <= pools_.size() );
	assert( pools_.size() > 0 );
	assert( numPools + startPool <= pools_[0].size() );
	values.resize( 4 + numVoxels * numPools );

	for ( unsigned int i = 0; i < numVoxels; ++i ) {
		const double* v = pools_[ startVoxel + i ].S();
		for ( unsigned int j = 0; j < numPools; ++j ) {
			values[ 4 + j * numVoxels + i]  = v[ j + startPool ];
		}
	}
}

void Ksolve::setBlock( const vector< double >& values )
{
	unsigned int startVoxel = values[0];
	unsigned int numVoxels = values[1];
	unsigned int startPool = values[2];
	unsigned int numPools = values[3];

	assert( startVoxel >= startVoxel_ );
	assert( numVoxels <= pools_.size() );
	assert( pools_.size() > 0 );
	assert( numPools + startPool <= pools_[0].size() );

	for ( unsigned int i = 0; i < numVoxels; ++i ) {
		double* v = pools_[ startVoxel + i ].varS();
		for ( unsigned int j = 0; j < numPools; ++j ) {
			v[ j + startPool ] = values[ 4 + j * numVoxels + i ];
		}
	}
}
