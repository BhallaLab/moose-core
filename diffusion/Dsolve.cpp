/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2014 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
#include "SparseMatrix.h"
#include "KinSparseMatrix.h"
#include "ZombiePoolInterface.h"
#include "DiffPoolVec.h"
#include "FastMatrixElim.h"
#include "Dsolve.h"
#include "../mesh/Boundary.h"
#include "../mesh/MeshEntry.h"
#include "../mesh/ChemCompt.h"
#include "../mesh/MeshCompt.h"

const Cinfo* Dsolve::initCinfo()
{
		///////////////////////////////////////////////////////
		// Field definitions
		///////////////////////////////////////////////////////
		
		static ValueFinfo< Dsolve, Id > stoich (
			"stoich",
			"Stoichiometry object for handling this reaction system.",
			&Dsolve::setStoich,
			&Dsolve::getStoich
		);

		static ReadOnlyValueFinfo< Dsolve, unsigned int > numVoxels(
			"numVoxels",
			"Number of voxels in the core reac-diff system, on the "
			"current diffusion solver. ",
			&Dsolve::getNumVoxels
		);
		static LookupValueFinfo< 
				Dsolve, unsigned int, vector< double > > nVec(
			"nVec",
			"vector of # of molecules along diffusion length, "
			"looked up by pool index",
			&Dsolve::setNvec,
			&Dsolve::getNvec
		);

		static ValueFinfo< Dsolve, unsigned int > numPools(
			"numPools",
			"Number of molecular pools in the entire reac-diff system, "
			"including variable, function and buffered.",
			&Dsolve::setNumPools,
			&Dsolve::getNumPools
		);

		static ValueFinfo< Dsolve, Id > compartment (
			"compartment",
			"Reac-diff compartment in which this diffusion system is "
			"embedded.",
			&Dsolve::setCompartment,
			&Dsolve::getCompartment
		);


		///////////////////////////////////////////////////////
		// DestFinfo definitions
		///////////////////////////////////////////////////////

		static DestFinfo process( "process",
			"Handles process call",
			new ProcOpFunc< Dsolve >( &Dsolve::process ) );
		static DestFinfo reinit( "reinit",
			"Handles reinit call",
			new ProcOpFunc< Dsolve >( &Dsolve::reinit ) );
		
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

	static Finfo* dsolveFinfos[] =
	{
		&stoich,			// Value
		&compartment,		// Value
		&numVoxels,			// ReadOnlyValue
		&nVec,				// LookupValue
		&numPools,			// Value
		&proc,				// SharedFinfo
	};
	
	static Dinfo< Dsolve > dinfo;
	static  Cinfo ksolveCinfo(
		"Dsolve",
		Neutral::initCinfo(),
		dsolveFinfos,
		sizeof(dsolveFinfos)/sizeof(Finfo *),
		&dinfo
	);

	return &ksolveCinfo;
}

static const Cinfo* ksolveCinfo = Dsolve::initCinfo();

//////////////////////////////////////////////////////////////
// Class definitions
//////////////////////////////////////////////////////////////
Dsolve::Dsolve()
	: numTotPools_( 0 ),
		numLocalPools_( 0 ),
		poolStartIndex_( 0 ),
		numVoxels_( 0 )
{;}

Dsolve::~Dsolve()
{;}

//////////////////////////////////////////////////////////////
// Field access functions
//////////////////////////////////////////////////////////////

void Dsolve::setNvec( unsigned int pool, vector< double > vec )
{
	if ( pool < pools_.size() ) {
		if ( vec.size() != pools_[pool].getNumVoxels() ) {
			cout << "Warning: Dsolve::setNvec: pool index out of range\n";
		} else {
			pools_[ pool ].setNvec( vec );
		}
	}
}

vector< double > Dsolve::getNvec( unsigned int pool ) const
{
	static vector< double > ret;
	if ( pool <  pools_.size() )
		return pools_[pool].getNvec();

	cout << "Warning: Dsolve::setNvec: pool index out of range\n";
	return ret;
}

//////////////////////////////////////////////////////////////
// Process operations.
//////////////////////////////////////////////////////////////
void Dsolve::process( const Eref& e, ProcPtr p )
{
	for ( vector< DiffPoolVec >::iterator 
					i = pools_.begin(); i != pools_.end(); ++i ) {
		i->advance( p->dt );
	}
}

void Dsolve::reinit( const Eref& e, ProcPtr p )
{
	build( p->dt );
	for ( vector< DiffPoolVec >::iterator 
					i = pools_.begin(); i != pools_.end(); ++i ) {
		i->reinit();
	}
}
//////////////////////////////////////////////////////////////
// Solver coordination and setup functions
//////////////////////////////////////////////////////////////

void Dsolve::setStoich( Id id )
{
	stoich_ = id; 
}

Id Dsolve::getStoich() const
{
	return stoich_;
}

void Dsolve::setCompartment( Id id )
{
	const Cinfo* c = id.element()->cinfo();
	if ( c->isA( "NeuroMesh" ) || c->isA( "CylMesh" ) ) {
		compartment_ = id;
		numVoxels_ = Field< unsigned int >::get( id, "numDiffCompts" );
		/*
		const MeshCompt* m = reinterpret_cast< const MeshCompt* >( 
						id.eref().data() );
		numVoxels_ = m->getStencil().nRows();
		*/
	} else {
		cout << "Warning: Dsolve::setCompartment:: compartment must be "
				"NeuroMesh or CylMesh, you tried :" << c->name() << endl;
	}
}

Id Dsolve::getCompartment() const
{
	return compartment_;
}
/////////////////////////////////////////////////////////////
// Solver building
//////////////////////////////////////////////////////////////

// Happens at reinit, long after all pools are built.
// By this point the diffusion consts etc will be assigned to the
// poolVecs.
// This requires
// - Stoich should be assigned
// - compartment should be assigned so we know how many voxels.
// - Stoich should have had the path set so we know numPools. It needs
// 		to know the numVoxels from the compartment. AT the time of
// 		path setting the zombification is done, which takes the Id of
// 		the solver.
// - After this build can be done. Just reinit doesn't make sense since
// 		the build does a lot of things which should not be repeated for
// 		each reinit.
void Dsolve::build( double dt )
{
	const MeshCompt* m = reinterpret_cast< const MeshCompt* >( 
						compartment_.eref().data() );
	// For now start with local pools only.
	if ( stoich_ != Id() )
		numLocalPools_ = Field< unsigned int >::get( stoich_, "numAllPools" );
	else
		numLocalPools_ = 1;
	pools_.resize( numLocalPools_ );

	for ( unsigned int i = 0; i < numLocalPools_; ++i ) {
		FastMatrixElim elim( m->getStencil() );
		vector< unsigned int > parentVoxel = m->getParentVoxel();
		elim.setDiffusionAndTransport( parentVoxel,
			pools_[i].getDiffConst(), pools_[i].getMotorConst(), dt );
		elim.hinesReorder( parentVoxel );
		vector< unsigned int > diagIndex;
		vector< double > diagVal;
		vector< Triplet< double > > fops;

		pools_[i].setNumVoxels( numVoxels_ );
		elim.buildForwardElim( diagIndex, fops );
		elim.buildBackwardSub( diagIndex, fops, diagVal );
		pools_[i].setOps( fops, diagVal );
	}
}

/////////////////////////////////////////////////////////////
// Zombie Pool Access functions
//////////////////////////////////////////////////////////////
//
unsigned int Dsolve::getNumVarPools() const
{
	return 0;
}

unsigned int Dsolve::getNumVoxels() const
{
	return numVoxels_;
}

unsigned int Dsolve::convertIdToPoolIndex( const Eref& e ) const
{
	unsigned int i  = e.id().value() - poolMapStart_;
	if ( i < poolMap_.size() ) {
		return poolMap_[i];
	}
	cout << "Warning: Dsolve::convertIdToPoollndex: Id out of range, (" <<
		poolMapStart_ << ", " << e.id() << ", " <<
		poolMap_.size() + poolMapStart_ << "\n";
	return 0;
}

void Dsolve::setN( const Eref& e, double v )
{
	unsigned int vox = e.dataIndex();
	if ( vox < numVoxels_ )
		pools_[ convertIdToPoolIndex( e ) ].setN( vox, v );
	else 
		cout << "Warning: Dsolve::setN: Eref out of range\n";
}

double Dsolve::getN( const Eref& e ) const
{
	unsigned int vox = e.dataIndex();
	if ( vox <  numVoxels_ )
		return pools_[ convertIdToPoolIndex( e ) ].getN( vox );
	cout << "Warning: Dsolve::getN: Eref out of range\n";
	return 0.0;
}

void Dsolve::setNinit( const Eref& e, double v )
{
	unsigned int vox = e.dataIndex();
	if ( vox < numVoxels_ )
		pools_[ convertIdToPoolIndex( e ) ].setNinit( vox, v );
	else 
		cout << "Warning: Dsolve::setNinit: Eref out of range\n";
}

double Dsolve::getNinit( const Eref& e ) const
{
	unsigned int vox = e.dataIndex();
	if ( vox < numVoxels_ )
		return pools_[ convertIdToPoolIndex( e ) ].getNinit( vox );
	cout << "Warning: Dsolve::getNinit: Eref out of range\n";
	return 0.0;
}

void Dsolve::setDiffConst( const Eref& e, double v )
{
	pools_[ convertIdToPoolIndex( e ) ].setDiffConst( v );
}

double Dsolve::getDiffConst( const Eref& e ) const
{
	return pools_[ convertIdToPoolIndex( e ) ].getDiffConst();
}

void Dsolve::setNumPools( unsigned int numPoolSpecies )
{
	// Decompose numPoolSpecies here, assigning some to each node.
	numTotPools_ = numPoolSpecies;
	numLocalPools_ = numPoolSpecies;
	poolStartIndex_ = 0;

	pools_.resize( numLocalPools_ );
	for ( unsigned int i = 0 ; i < numLocalPools_; ++i ) {
		pools_[i].setNumVoxels( numVoxels_ );
		// pools_[i].setId( reversePoolMap_[i] );
		// pools_[i].setParent( me );
	}
}

unsigned int Dsolve::getNumPools() const
{
	return numTotPools_;
}
