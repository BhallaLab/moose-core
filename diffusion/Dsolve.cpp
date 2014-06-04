/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2014 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
#include "ElementValueFinfo.h"
#include "SparseMatrix.h"
#include "KinSparseMatrix.h"
#include "ZombiePoolInterface.h"
#include "DiffPoolVec.h"
#include "FastMatrixElim.h"
#include "../mesh/VoxelJunction.h"
#include "DiffJunction.h"
#include "Dsolve.h"
#include "../mesh/Boundary.h"
#include "../mesh/MeshEntry.h"
#include "../mesh/ChemCompt.h"
#include "../mesh/MeshCompt.h"
#include "../shell/Wildcard.h"
#include "../kinetics/PoolBase.h"
#include "../kinetics/Pool.h"
#include "../kinetics/BufPool.h"
#include "../kinetics/FuncPool.h"
#include "../ksolve/ZombiePool.h"
#include "../ksolve/ZombieBufPool.h"
#include "../ksolve/ZombieFuncPool.h"

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
		
		static ElementValueFinfo< Dsolve, string > path (
			"path",
			"Path of reaction system. Must include all the pools that "
			"are to be handled by the Dsolve, can also include other "
			"random objects, which will be ignored.",
			&Dsolve::setPath,
			&Dsolve::getPath
		);

		static ReadOnlyValueFinfo< Dsolve, unsigned int > numVoxels(
			"numVoxels",
			"Number of voxels in the core reac-diff system, on the "
			"current diffusion solver. ",
			&Dsolve::getNumVoxels
		);
		static ReadOnlyValueFinfo< Dsolve, unsigned int > numAllVoxels(
			"numAllVoxels",
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

		static DestFinfo buildNeuroMeshJunctions( "buildNeuroMeshJunctions",
			"Builds junctions between NeuroMesh, SpineMesh and PsdMesh",
			new EpFunc2< Dsolve, Id, Id >( 
					&Dsolve::buildNeuroMeshJunctions ) );
		
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
		&stoich,			// ElementValue
		&path,				// ElementValue
		&compartment,		// Value
		&numVoxels,			// ReadOnlyValue
		&numAllVoxels,			// ReadOnlyValue
		&nVec,				// LookupValue
		&numPools,			// Value
		&buildNeuroMeshJunctions, 	// DestFinfo
		&proc,				// SharedFinfo
	};
	
	static Dinfo< Dsolve > dinfo;
	static  Cinfo dsolveCinfo(
		"Dsolve",
		Neutral::initCinfo(),
		dsolveFinfos,
		sizeof(dsolveFinfos)/sizeof(Finfo *),
		&dinfo
	);

	return &dsolveCinfo;
}

static const Cinfo* dsolveCinfo = Dsolve::initCinfo();

//////////////////////////////////////////////////////////////
// Class definitions
//////////////////////////////////////////////////////////////
Dsolve::Dsolve()
	: 
		dt_( -1.0 ),
		numTotPools_( 0 ),
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

static double integ( double myN, double rf, double rb, double dt )
{
	const double EPSILON = 1e-12;
	if ( myN > EPSILON && rf > EPSILON ) {
		double C = exp( -rf * dt / myN );
		myN *= C + ( rb / rf ) * ( 1.0 - C );
	} else {
		myN += ( rb - rf ) * dt;
	}
	if ( myN < 0.0 )
		return 0.0;
	return myN;
}

/**
 * Computes flux through a junction between diffusion solvers.
 * Most used at junctions on spines and PSDs, but can also be used
 * when a given diff solver is decomposed. At present the lookups
 * on the other diffusion solver assume that the data is on the local
 * node. Once this works well I can figure out how to do across nodes.
 */
void Dsolve::calcJunction( const DiffJunction& jn, double dt )
{
	const double EPSILON = 1e-15;
	Id oid( jn.otherDsolve );
	assert ( oid != Id() );
	assert ( oid.element()->cinfo()->isA( "Dsolve" ) );

	Dsolve* other = reinterpret_cast< Dsolve* >( oid.eref().data() );

	assert( jn.otherPools.size() == jn.myPools.size() );
	for ( unsigned int i = 0; i < jn.myPools.size(); ++i ) {
		DiffPoolVec& myDv = pools_[ jn.myPools[i] ];
		if ( myDv.getDiffConst() < EPSILON )
			continue;
		DiffPoolVec& otherDv = other->pools_[ jn.otherPools[i] ];
		for ( vector< VoxelJunction >::const_iterator
			j = jn.vj.begin(); j != jn.vj.end(); ++j ) {
			double myN = myDv.getN( j->first );
			double otherN = otherDv.getN( j->second );
			// Here we do an exp Euler calculation
			// rf is rate from self to other.
			double k = myDv.getDiffConst() * j->diffScale; 
			double lastN = myN;
			myN = integ( myN, 
				k * myN / j->firstVol, 
				k * otherN / j->secondVol, 
				dt 
			);
			otherN += lastN - myN; // Simple mass conservation
			if ( otherN < 0.0 ) { // Avoid negatives
				myN += otherN;
				otherN = 0.0;
			}
			myDv.setN( j->first, myN );
			otherDv.setN( j->second, otherN );
		}
	}
}

void Dsolve::process( const Eref& e, ProcPtr p )
{
	for ( vector< DiffPoolVec >::iterator 
					i = pools_.begin(); i != pools_.end(); ++i ) {
		i->advance( p->dt );
	}

	for ( vector< DiffJunction >::const_iterator
			i = junctions_.begin(); i != junctions_.end(); ++i ) {
		calcJunction( *i, p->dt );
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
	if ( !id.element()->cinfo()->isA( "Stoich" ) ) {
		cout << "Dsolve::setStoich::( " << id << " ): Error: provided Id is not a Stoich\n";
		return;
	}

	stoich_ = id;
	poolMap_ = Field< vector< unsigned int > >::get( stoich_, "poolIdMap" );
	poolMapStart_ = poolMap_.back();
	poolMap_.pop_back();

	path_ = Field< string >::get( stoich_, "path" );

	for ( unsigned int i = 0; i < poolMap_.size(); ++i ) {
		if ( poolMap_[i] != ~0U ) {
			Id pid( i + poolMapStart_ );
			assert( pid.element()->cinfo()->isA( "PoolBase" ) );
			PoolBase* pb = 
					reinterpret_cast< PoolBase* >( pid.eref().data());
			double diffConst = pb->getDiffConst( pid.eref() );
			double motorConst = pb->getMotorConst( pid.eref() );
			pools_[ poolMap_[i] ].setId( pid.value() );
			pools_[ poolMap_[i] ].setDiffConst( diffConst );
			pools_[ poolMap_[i] ].setMotorConst( motorConst );
		}
	}
}

Id Dsolve::getStoich() const
{
	return stoich_;
}

/// Inherited, defining dummy function here.
void Dsolve::setDsolve( Id dsolve )
{;}

void Dsolve::setCompartment( Id id )
{
	const Cinfo* c = id.element()->cinfo();
	if ( c->isA( "NeuroMesh" ) || c->isA( "SpineMesh" ) || 
					c->isA( "PsdMesh" ) || c->isA( "CylMesh" ) ) {
		compartment_ = id;
		numVoxels_ = Field< unsigned int >::get( id, "numMesh" );
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

void Dsolve::makePoolMapFromElist( const vector< ObjId >& elist, 
				vector< Id >& temp )
{
	unsigned int minId;
	unsigned int maxId;
	temp.resize( 0 );
	for ( vector< ObjId >::const_iterator 
			i = elist.begin(); i != elist.end(); ++i ) {
		if ( i->element()->cinfo()->isA( "PoolBase" ) ) {
			temp.push_back( i->id );
			if ( minId == 0 ) 
				maxId = minId = i->id.value();
			else if ( i->id.value() < minId )
				minId = i->id.value();
			else if ( i->id.value() > maxId )
				maxId = i->id.value();
		}
	}

	if ( temp.size() == 0 ) {
		cout << "Dsolve::makePoolMapFromElist::( " << path_ << 
				" ): Error: path is has no pools\n";
		return;
	}

	stoich_ = Id();
	poolMapStart_ = minId;
	poolMap_.resize( 1 + maxId - minId );
	for ( unsigned int i = 0; i < temp.size(); ++i ) {
		unsigned int idValue = temp[i].value();
		assert( idValue >= minId );
		assert( idValue - minId < poolMap_.size() );
		poolMap_[ idValue - minId ] = i;
	}
}

void Dsolve::setPath( const Eref& e, string path )
{
	vector< ObjId > elist;
	simpleWildcardFind( path, elist );
	if ( elist.size() == 0 ) {
		cout << "Dsolve::setPath::( " << path << " ): Error: path is empty\n";
		return;
	}
	vector< Id > temp;
	makePoolMapFromElist( elist, temp );

	setNumPools( temp.size() );
	for ( unsigned int i = 0; i < temp.size(); ++i ) {
	 	Id id = temp[i];
		double diffConst = Field< double >::get( id, "diffConst" );
		double motorConst = Field< double >::get( id, "motorConst" );
		const Cinfo* c = id.element()->cinfo();
		if ( c == Pool::initCinfo() )
			PoolBase::zombify( id.element(), ZombiePool::initCinfo(), Id(), e.id() );
		else if ( c == BufPool::initCinfo() )
			PoolBase::zombify( id.element(), ZombieBufPool::initCinfo(), Id(), e.id() );
		else if ( c == FuncPool::initCinfo() )
			PoolBase::zombify( id.element(), ZombieFuncPool::initCinfo(), Id(), e.id() );
		else
			cout << "Error: Dsolve::setPath( " << path << " ): unknown pool class:" << c->name() << endl; 
		id.element()->resize( numVoxels_ );

		unsigned int j = temp[i].value() - poolMapStart_;
		assert( j < poolMap_.size() );
		pools_[ poolMap_[i] ].setId( id.value() );
		pools_[ poolMap_[j] ].setDiffConst( diffConst );
		pools_[ poolMap_[j] ].setMotorConst( motorConst );
	}
}

string Dsolve::getPath( const Eref& e ) const
{
	return path_;
}

/////////////////////////////////////////////////////////////
// Solver building
//////////////////////////////////////////////////////////////

/** 
 * build: This function is called either by setStoich or setPath.
 * By this point the diffusion consts etc will be assigned to the
 * poolVecs.
 * This requires
 * - Stoich should be assigned, OR
 * - A 'path' should be assigned which has been traversed to find pools.
 * - compartment should be assigned so we know how many voxels.
 * - If Stoich, its 'path' should be set so we know numPools. It needs
 * to know the numVoxels from the compartment. At the time of
 * path setting the zombification is done, which takes the Id of
 * the solver.
 * - After this build can be done. Just reinit doesn't make sense since
 * the build does a lot of things which should not be repeated for
 * each reinit.
 */

void Dsolve::build( double dt )
{
	if ( doubleEq( dt, dt_ ) )
		return;
	dt_ = dt;

	const MeshCompt* m = reinterpret_cast< const MeshCompt* >( 
						compartment_.eref().data() );
	unsigned int numVoxels = m->getNumEntries();

	for ( unsigned int i = 0; i < numLocalPools_; ++i ) {
		bool debugFlag = false;
		vector< unsigned int > diagIndex;
		vector< double > diagVal;
		vector< Triplet< double > > fops;
		FastMatrixElim elim( numVoxels, numVoxels );
		if ( elim.buildForDiffusion( 
			m->getParentVoxel(), m->getVoxelVolume(), 
			m->getVoxelArea(), m->getVoxelLength(), 
		    pools_[i].getDiffConst(), pools_[i].getMotorConst(), dt ) ) 
		{
			vector< unsigned int > parentVoxel = m->getParentVoxel();
			assert( elim.checkSymmetricShape() );
			vector< unsigned int > lookupOldRowsFromNew;
			elim.hinesReorder( parentVoxel, lookupOldRowsFromNew );
			assert( elim.checkSymmetricShape() );
			pools_[i].setNumVoxels( numVoxels_ );
			elim.buildForwardElim( diagIndex, fops );
			elim.buildBackwardSub( diagIndex, fops, diagVal );
			elim.opsReorder( lookupOldRowsFromNew, fops, diagVal );
			if (debugFlag )
				elim.print();
		}
		pools_[i].setOps( fops, diagVal );
	}
}

/**
 * Should be called only from the Dsolve handling the NeuroMesh.
 */
// Would like to permit vectors of spines and psd compartments.
void Dsolve::buildNeuroMeshJunctions( const Eref& e, Id spineD, Id psdD )
{
	if ( !compartment_.element()->cinfo()->isA( "NeuroMesh" ) ) {
		cout << "Warning: Dsolve::buildNeuroMeshJunction: Compartment '" <<
				compartment_.path() << "' is not a NeuroMesh\n";
		return;
	}
	Id spineMesh = Field< Id >::get( spineD, "compartment" );
	if ( !spineMesh.element()->cinfo()->isA( "SpineMesh" ) ) {
		cout << "Warning: Dsolve::buildNeuroMeshJunction: Compartment '" <<
				spineMesh.path() << "' is not a SpineMesh\n";
		return;
	}
	Id psdMesh = Field< Id >::get( psdD, "compartment" );
	if ( !psdMesh.element()->cinfo()->isA( "PsdMesh" ) ) {
		cout << "Warning: Dsolve::buildNeuroMeshJunction: Compartment '" <<
				psdMesh.path() << "' is not a PsdMesh\n";
		return;
	}

	buildMeshJunctions( spineD, e.id() );
	buildMeshJunctions( psdD, spineD );
}

// Static utility func for building junctions
void Dsolve::buildMeshJunctions( Id self, Id other )
{
	DiffJunction jn; // This is based on the Spine Dsolver.
	jn.otherDsolve = other.value();
	// Map pools between Dsolves
	Dsolve* mySolve = reinterpret_cast< Dsolve* >( self.eref().data() );
	map< string, unsigned int > myPools;
	for ( unsigned int i = 0; i < mySolve->pools_.size(); ++i ) {
			Id pool( mySolve->pools_[i].getId() );
			assert( pool != Id() );
			myPools[ pool.element()->getName() ] = i;
	}

	const Dsolve* otherSolve = reinterpret_cast< const Dsolve* >(
					other.eref().data() );
	for ( unsigned int i = 0; i < otherSolve->pools_.size(); ++i ) {
		Id otherPool( otherSolve->pools_[i].getId() );
		map< string, unsigned int >::iterator p = 
			myPools.find( otherPool.element()->getName() );
		if ( p != myPools.end() ) {
			jn.otherPools.push_back( i );
			jn.myPools.push_back( p->second );
		}
	}

	// Map voxels between meshes.
	Id myMesh = Field< Id >::get( self, "compartment" );
	Id otherMesh = Field< Id >::get( other, "compartment" );

	const ChemCompt* myCompt = reinterpret_cast< const ChemCompt* >( 
					myMesh.eref().data() );
	const ChemCompt* otherCompt = reinterpret_cast< const ChemCompt* >( 
					otherMesh.eref().data() );
	myCompt->matchMeshEntries( otherCompt, jn.vj );
	vector< double > myVols = myCompt->getVoxelVolume();
	vector< double > otherVols = otherCompt->getVoxelVolume();
	for ( vector< VoxelJunction >::iterator 
		i = jn.vj.begin(); i != jn.vj.end(); ++i ) {
		i->firstVol = myVols[i->first];
		i->secondVol = otherVols[i->second];
	}

	mySolve->junctions_.push_back( jn );
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

void Dsolve::setMotorConst( const Eref& e, double v )
{
	pools_[ convertIdToPoolIndex( e ) ].setMotorConst( v );
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

void Dsolve::getBlock( vector< double >& values ) const
{
	unsigned int startVoxel = values[0];
	unsigned int numVoxels = values[1];
	unsigned int startPool = values[2];
	unsigned int numPools = values[3];

	assert( startVoxel + numVoxels <= numVoxels_ );
	assert( startPool >= poolStartIndex_ );
	assert( numPools + startPool <= numLocalPools_ );
	values.resize( 4 );

	for ( unsigned int i = 0; i < numPools; ++i ) {
		unsigned int j = i + startPool;
		if ( j >= poolStartIndex_ && j < poolStartIndex_ + numLocalPools_ ){
			vector< double >::const_iterator q =
				pools_[ j - poolStartIndex_ ].getNvec().begin();
				
			values.insert( values.end(),
				q + startVoxel, q + startVoxel + numVoxels );
		}
	}
}

void Dsolve::setBlock( const vector< double >& values )
{
	unsigned int startVoxel = values[0];
	unsigned int numVoxels = values[1];
	unsigned int startPool = values[2];
	unsigned int numPools = values[3];

	assert( startVoxel + numVoxels <= numVoxels_ );
	assert( startPool >= poolStartIndex_ );
	assert( numPools + startPool <= numLocalPools_ );

	for ( unsigned int i = 0; i < numPools; ++i ) {
		unsigned int j = i + startPool;
		if ( j >= poolStartIndex_ && j < poolStartIndex_ + numLocalPools_ ){
			vector< double >::const_iterator 
				q = values.begin() + 4 + i * numVoxels;
			pools_[ j - poolStartIndex_ ].setNvec( startVoxel, numVoxels, q );
		}
	}
}
