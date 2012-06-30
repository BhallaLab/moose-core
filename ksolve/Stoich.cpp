/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "StoichHeaders.h"
#include "ElementValueFinfo.h"
#include "Pool.h"
#include "BufPool.h"
#include "FuncPool.h"
#include "Reac.h"
#include "Enz.h"
#include "MMenz.h"
#include "SumFunc.h"
#include "MathFunc.h"
#include "Boundary.h"
#include "MeshEntry.h"
#include "ChemMesh.h"
#include "ZombiePool.h"
#include "ZombieBufPool.h"
#include "ZombieFuncPool.h"
#include "ZombieReac.h"
#include "ZombieEnz.h"
#include "ZombieMMenz.h"
#include "ZombieSumFunc.h"
#include "../mesh/Stencil.h"

#include "../shell/Shell.h"

#ifdef USE_GSL
#include <gsl/gsl_errno.h>
#endif

#define EPSILON 1e-15

static SrcFinfo1< Id >* plugin()
{
	static SrcFinfo1< Id > ret(
		"plugin", 
		"Sends out Stoich Id so that plugins can directly access fields and functions"
	);
	return &ret;
}

static SrcFinfo3< unsigned int, vector< unsigned int >, vector< double > >* nodeDiffBoundary()
{
	static SrcFinfo3< unsigned int, vector< unsigned int >, vector< double > > nodeDiffBoundary(
		"nodeDiffBoundary", 
		"Sends mol #s across boundary between nodes, to calculate diffusion"
		"terms. arg1 is originating node, arg2 is list of meshIndices for"
		"which data is being transferred, and arg3 are the 'n' values for"
		"all the pools on the specified meshIndices, to be plugged into"
		"the appropriate place on the recipient node's S_ matrix"
	);
	return &nodeDiffBoundary;
}


const Cinfo* Stoich::initCinfo()
{
		//////////////////////////////////////////////////////////////
		// Field Definitions
		//////////////////////////////////////////////////////////////
		static ValueFinfo< Stoich, bool > useOneWay(
			"useOneWayReacs",
			"Flag: use bidirectional or one-way reacs. One-way is needed"
			"for Gillespie type stochastic calculations. Two-way is"
			"likely to be margninally more efficient in ODE calculations",
			&Stoich::setOneWay,
			&Stoich::getOneWay
		);

		static ReadOnlyValueFinfo< Stoich, unsigned int > nVarPools(
			"nVarPools",
			"Number of variable molecule pools in the reac system",
			&Stoich::getNumVarPools
		);

		/*
		static LookupValueFinfo< Stoich, short, double > compartmentVolume(
			"compartmentVolume",
			"Size of specified compartment",
			&Stoich::setCompartmentVolume,
			&Stoich::getCompartmentVolume
		);
		*/

		static ElementValueFinfo< Stoich, string > path(
			"path",
			"Path of reaction system to take over",
			&Stoich::setPath,
			&Stoich::getPath
		);

		static ReadOnlyValueFinfo< Stoich, unsigned int > numMeshEntries(
			"numMeshEntries",
			"Number of meshEntries in reac-diff system",
			&Stoich::getNumMeshEntries
		);

		static ReadOnlyValueFinfo< Stoich, double > estimatedDt(
			"estimatedDt",
			"Estimate of fastest (smallest) timescale in system."
			"This is fallible because it depends on instantaneous concs,"
			"which of course change over the course of the simulation.",
			&Stoich::getEstimatedDt
		);

		//////////////////////////////////////////////////////////////
		// MsgDest Definitions
		//////////////////////////////////////////////////////////////
		/*
		static DestFinfo process( "process",
			"Handles process call",
			new ProcOpFunc< Stoich >( &Stoich::process ) );
		static DestFinfo reinit( "reinit",
			"Handles reinint call",
			new ProcOpFunc< Stoich >( &Stoich::reinit ) );
			*/

		static DestFinfo meshSplit( "meshSplit",
			"Handles message from ChemMesh that defines how "
			"meshEntries are decomposed on this node, and how they "
			"communicate between nodes."
			"Args: (volumeVectorForAllEntries, localEntryList, "
			"outgoingDiffusion[node#][entry#], "
			"incomingDiffusion[node#][entry#])",
			new OpFunc4< Stoich, 
				vector< double >, vector< unsigned int >,
				vector< vector< unsigned int > >,
				vector< vector< unsigned int > >
			>( &Stoich::meshSplit )
		);

		//////////////////////////////////////////////////////////////
		// FieldElementFinfo defintion for Ports. Assume up to 16.
		//////////////////////////////////////////////////////////////
		static FieldElementFinfo< Stoich, Port > portFinfo( "port",
			"Sets up field Elements for ports",
			Port::initCinfo(),
			&Stoich::getPort,
			&Stoich::setNumPorts,
			&Stoich::getNumPorts,
			16
		);

		//////////////////////////////////////////////////////////////
		// SharedMsg Definitions
		//////////////////////////////////////////////////////////////
		/*
		static Finfo* procShared[] = {
			&process, &reinit
		};
		static SharedFinfo proc( "proc",
			"Shared message for process and reinit",
			procShared, sizeof( procShared ) / sizeof( const Finfo* )
		);
		*/

	static Finfo* stoichFinfos[] = {
		&useOneWay,		// Value
		&nVarPools,		// Value
		&numMeshEntries,		// Value
		&estimatedDt,		// ReadOnlyValue
		// &compartmentVolume,	//Value
		&path,			// Value
		plugin(),		// SrcFinfo
		nodeDiffBoundary(),		// SrcFinfo
		&meshSplit,		// DestFinfo
		&portFinfo,		// FieldElementFinfo
	};

	static Cinfo stoichCinfo (
		"Stoich",
		Neutral::initCinfo(),
		stoichFinfos,
		sizeof( stoichFinfos ) / sizeof ( Finfo* ),
		new Dinfo< Stoich >()
	);

	return &stoichCinfo;
}

//////////////////////////////////////////////////////////////
// Class definitions
//////////////////////////////////////////////////////////////
static const Cinfo* stoichCinfo = Stoich::initCinfo();

Stoich::Stoich()
	: 
		useOneWay_( 0 ),
		path_( "" ),
		S_(1),
		Sinit_(1),
		y_(1),
		localMeshEntries_( 1 ),
		totPortSize_( 0 ),
		objMapStart_( 0 ),
		numVarPools_( 0 ),
		numVarPoolsBytes_( 0 ),
		numReac_( 0 )
{;}

Stoich::~Stoich()
{
	for ( vector< RateTerm* >::iterator i = rates_.begin();
		i != rates_.end(); ++i )
		delete *i;

	for ( vector< FuncTerm* >::iterator i = funcs_.begin();
		i != funcs_.end(); ++i )
		delete *i;
}

//////////////////////////////////////////////////////////////
// MsgDest Definitions
//////////////////////////////////////////////////////////////

// This must only be called by the object that is actually
// handling the processing: GssaStoich or GslIntegrator at this
// time. That is because this function may reallocate memory
// and its values must propagate serially to the calling object.
// Somewhat inefficient: Must be called on thread 0 but would ideally
// like to split among threads.
void Stoich::innerReinit()
{
	// This might not be true, since S_ maintains extra info for the 
	// off-node pools to which it connects.
	// assert( y_.size() == S_.size() );

	assert( y_.size() == localMeshEntries_.size() );
	assert( Sinit_.size() == S_.size() );

	for ( unsigned int i = 0; i < y_.size(); ++i ) {
		y_[i].assign( Sinit_[i].begin(), Sinit_[i].begin() + numVarPools_ );
	}
	S_ = Sinit_;

	for ( unsigned int i = 0; i < localMeshEntries_.size(); ++i ) {
		updateFuncs( 0, localMeshEntries_[i] );
	}
	// updateV( 0 );
}

/**
 * Handles incoming messages representing influx of molecules
 */
void Stoich::influx( unsigned int port, vector< double > pool )
{
	/*
	assert( pool.size() == inPortEnd_ - inPortStart_ );
	unsigned int j = 0;
	for ( unsigned int i = inPortStart_; i < inPortEnd_; ++i ) {
		S_[i] += pool[j++];
	}
	*/
}

/**
 * Should really do this using a map indexed by SpeciesId.
 */
void Stoich::handleAvailableMolsAtPort( unsigned int port, vector< SpeciesId > mols )
{
	/*
	vector< SpeciesId > ret;
	assert( port.field() < ports_.size() );
	ports_[port.field()]->findMatchingMolSpecies( molSpecies, ret );
	Port& p = ports_[ port.field() ];
	for ( vector< SpeciesId >::iterator i = species_.begin(); 
		i != species_.end(); ++i ) {
		if ( *i != DefaultSpeciesId ) {
			if ( p.availableMols_.find( *i ) != p.availableMols_.end() ) {
				ret.push_back( *i );
				p->usedMols_.push_back( i->second );
			}
		}
	}
	*/
}

void Stoich::handleMatchedMolsAtPort( unsigned int port, vector< SpeciesId > mols )
{
	;
}

//////////////////////////////////////////////////////////////
// Field Definitions
//////////////////////////////////////////////////////////////

void Stoich::setOneWay( bool v )
{
	useOneWay_ = v;
}

bool Stoich::getOneWay() const
{
	return useOneWay_;
}

void Stoich::setPath( const Eref& e, const Qinfo* q, string v )
{
	if ( path_ != "" && path_ != v ) {
		// unzombify( path_ );
		cout << "Stoich::setPath: need to clear old path.\n";
		return;
	}
	path_ = v;
	vector< Id > elist;
	Shell::wildcard( path_, elist );

	allocateObjMap( elist );
	allocateModel( elist );
	zombifyModel( e, elist );
	innerReinit();
	// y_.assign( Sinit_.begin(), Sinit_.begin() + numVarPools_ );
	// S_ = Sinit_;

	/*
	cout << "Zombified " << numVarPools_ << " Molecules, " <<
		numReac_ << " reactions\n";
	N_.print();
	*/
}

string Stoich::getPath( const Eref& e, const Qinfo* q ) const
{
	return path_;
}

unsigned int Stoich::getNumMeshEntries() const
{
	return localMeshEntries_.size();
}

double Stoich::getEstimatedDt() const
{
	return 1; // Dummy
}

unsigned int Stoich::getNumVarPools() const
{
	return numVarPools_;
}

Port* Stoich::getPort( unsigned int i )
{
	assert( i < ports_.size() );
	return &ports_[i];
}

unsigned int Stoich::getNumPorts() const
{
	return ports_.size();
}

void Stoich::setNumPorts( unsigned int num )
{
	assert( num < 10000 );
	ports_.resize( num );
}

//////////////////////////////////////////////////////////////
// Model zombification functions
//////////////////////////////////////////////////////////////
void Stoich::allocateObjMap( const vector< Id >& elist )
{
	objMapStart_ = ~0;
	unsigned int maxId = 0;
	for ( vector< Id >::const_iterator i = elist.begin(); i != elist.end(); ++i ){
		if ( objMapStart_ > i->value() )
			objMapStart_ = i->value();
		if ( maxId < i->value() )
			maxId = i->value();
	}
	objMap_.resize(0);
	objMap_.resize( 1 + maxId - objMapStart_, 0 );
	assert( objMap_.size() >= elist.size() );
}

void Stoich::allocateModel( const vector< Id >& elist )
{
	static const Cinfo* poolCinfo = Pool::initCinfo();
	static const Cinfo* bufPoolCinfo = BufPool::initCinfo();
	static const Cinfo* funcPoolCinfo = FuncPool::initCinfo();
	static const Cinfo* reacCinfo = Reac::initCinfo();
	static const Cinfo* enzCinfo = Enz::initCinfo();
	static const Cinfo* mmEnzCinfo = MMenz::initCinfo();
	static const Cinfo* sumFuncCinfo = SumFunc::initCinfo();
	// static const Cinfo* meshEntryCinfo = MeshEntry::initCinfo();
	numVarPools_ = 0;
	numReac_ = 0;
	vector< Id > bufPools;
	vector< Id > funcPools;
	unsigned int numFunc = 0;
	for ( vector< Id >::const_iterator i = elist.begin(); i != elist.end(); ++i ){
		Element* ei = (*i)();
		if ( ei->cinfo() == poolCinfo ) {
			objMap_[ i->value() - objMapStart_ ] = numVarPools_;
			idMap_.push_back( *i );
			++numVarPools_;
		} else if ( ei->cinfo() == bufPoolCinfo ) {
			bufPools.push_back( *i );
		} else if ( ei->cinfo() == funcPoolCinfo ) {
			funcPools.push_back( *i );
		} else if ( ei->cinfo() == mmEnzCinfo ){
			objMap_[ i->value() - objMapStart_ ] = numReac_;
			++numReac_;
		} else if ( ei->cinfo() == reacCinfo ) {
			if ( useOneWay_ ) {
				objMap_[ i->value() - objMapStart_ ] = numReac_;
				numReac_ += 2;
			} else {
				objMap_[ i->value() - objMapStart_ ] = numReac_;
				++numReac_;
			}
		} else if ( ei->cinfo() == enzCinfo ) {
			if ( useOneWay_ ) {
				objMap_[ i->value() - objMapStart_ ] = numReac_;
				numReac_ += 3;
			} else {
				objMap_[ i->value() - objMapStart_ ] = numReac_;
				numReac_ += 2;
			}
		} else if ( ei->cinfo() == sumFuncCinfo ){
			objMap_[ i->value() - objMapStart_ ] = numFunc;
			++numFunc;
		} 
	}

	numBufPools_ = 0;
	for ( vector< Id >::const_iterator i = bufPools.begin(); i != bufPools.end(); ++i ){
		objMap_[ i->value() - objMapStart_ ] = numVarPools_ + numBufPools_;
		idMap_.push_back( *i );
		++numBufPools_;
	}

	numFuncPools_ = numVarPools_ + numBufPools_;
	for ( vector< Id >::const_iterator i = funcPools.begin(); 
		i != funcPools.end(); ++i ) {
		objMap_[ i->value() - objMapStart_ ] = numFuncPools_++;
		idMap_.push_back( *i );
	}
	assert( idMap_.size() == numFuncPools_ );
	numFuncPools_ -= numVarPools_ + numBufPools_;
	assert( numFunc == numFuncPools_ );

	numVarPoolsBytes_ = numVarPools_ * sizeof( double );
	concInit_.resize( numVarPools_ + numBufPools_ + numFuncPools_, 0.0 );
	S_.resize( 1 );
	Sinit_.resize( 1 );
	y_.resize( 1 );
	flux_.resize( 1 );
	S_[0].resize( numVarPools_ + numBufPools_ + numFuncPools_, 0.0 );
	Sinit_[0].resize( numVarPools_ + numBufPools_ + numFuncPools_, 0.0);
	y_[0].resize( numVarPools_, 0.0 );
	flux_[0].resize( numVarPools_, 0.0 );

	diffConst_.resize( numVarPools_ + numBufPools_ + numFuncPools_, 0.0 );
	// compartment_.resize( numVarPools_ + numBufPools_ + numFuncPools_, 0 );
	species_.resize( numVarPools_ + numBufPools_ + numFuncPools_, 0 );
	rates_.resize( numReac_ );
	// v_.resize( numReac_, 0.0 ); // v is now allocated dynamically
	funcs_.resize( numFuncPools_ );
	N_.setSize( numVarPools_ + numBufPools_ + numFuncPools_, numReac_ );
}

void Stoich::zombifyModel( const Eref& e, const vector< Id >& elist )
{
	static const Cinfo* poolCinfo = Pool::initCinfo();
	static const Cinfo* bufPoolCinfo = BufPool::initCinfo();
	static const Cinfo* funcPoolCinfo = FuncPool::initCinfo();
	static const Cinfo* reacCinfo = Reac::initCinfo();
	static const Cinfo* enzCinfo = Enz::initCinfo();
	static const Cinfo* mmEnzCinfo = MMenz::initCinfo();
	// static const Cinfo* chemComptCinfo = ChemMesh::initCinfo();
	// static const Cinfo* sumFuncCinfo = SumFunc::initCinfo();
	// The FuncPool handles zombification of stuff coming in to it.
	vector< Id > meshEntries;

	for ( vector< Id >::const_iterator i = elist.begin(); i != elist.end(); ++i ){
		Element* ei = (*i)();
		if ( ei->cinfo() == poolCinfo ) {
			ZombiePool::zombify( e.element(), (*i)() );
		}
		else if ( ei->cinfo() == bufPoolCinfo ) {
			ZombieBufPool::zombify( e.element(), (*i)() );
		}
		else if ( ei->cinfo() == funcPoolCinfo ) {
			ZombieFuncPool::zombify( e.element(), (*i)() );
		}
		else if ( ei->cinfo() == reacCinfo ) {
			ZombieReac::zombify( e.element(), (*i)() );
		}
		else if ( ei->cinfo() == mmEnzCinfo ) {
			ZombieMMenz::zombify( e.element(), (*i)() );
		}
		else if ( ei->cinfo() == enzCinfo ) {
			ZombieEnz::zombify( e.element(), (*i)() );
		}
		/*
		else if ( ei->cinfo()->isA( "MeshEntry" ) ) {
			meshEntries.push_back( *i ); // Accumulate for later use. The
			// function needs all the pools to have been assimilated first.
		}
		*/
	}
	/*
	for ( vector< Id >::const_iterator i = meshEntries.begin(); i != meshEntries.end(); ++i ){
			zombifyChemMesh( *i ); // It retains its identity.
			// ZombieChemMesh::zombify( e.element(), (*i)() );
	}
	*/
}

void Stoich::handleRemesh( unsigned int numLocalMeshEntries, 
	vector< unsigned int > computedEntries, 
	vector< unsigned int > allocatedEntries, 
	vector< vector< unsigned int > > outgoingDiffusion, 
	vector< vector< unsigned int > > incomingDiffusion ) 
{
	cout << "Stoich::handleRemesh\n";
}

void Stoich::meshSplit(
	vector< double > vols,
	vector< unsigned int > localEntryList,
	vector< vector< unsigned int > > outgoingDiffusion,
	vector< vector< unsigned int > > incomingDiffusion
	)
{
	// cout << "Stoich::handleMeshSplit\n";
	unsigned int totalNumMeshEntries = vols.size();
	unsigned int numLocal = localEntryList.size();
	S_.resize( totalNumMeshEntries );
	Sinit_.resize( totalNumMeshEntries );
	y_.resize( numLocal );
	flux_.resize( numLocal );
	for ( unsigned int i = 0; i < numLocal; ++i ) {
		// Assume that these values will later be initialized
		unsigned int k = localEntryList[i]; // Converts to global index
		assert( k < totalNumMeshEntries );
		S_[ k ].resize( concInit_.size(), 0 );
		Sinit_[ k ].resize( concInit_.size(), 0 );
		for ( unsigned int j = 0; j < concInit_.size(); ++j ) {
			S_[k][j] = Sinit_[k][j] = concInit_[j] * vols[k] * NA;
		}
		y_[i].resize( numVarPools_, 0 );
		flux_[i].resize( numVarPools_, 0 );
	}
	localMeshEntries_ = localEntryList;
	outgoing_ = outgoingDiffusion;
	incoming_ = incomingDiffusion;
}

unsigned int Stoich::convertIdToPoolIndex( Id id ) const
{
	unsigned int i = id.value() - objMapStart_;
	assert( i < objMap_.size() );
	i = objMap_[i];
	assert( i < concInit_.size() );
	return i;
}

unsigned int Stoich::convertIdToReacIndex( Id id ) const
{
	unsigned int i = id.value() - objMapStart_;
	assert( i < objMap_.size() );
	i = objMap_[i];
	assert( i < rates_.size() );
	return i;
}

unsigned int Stoich::convertIdToFuncIndex( Id id ) const
{
	unsigned int i = id.value() - objMapStart_;
	assert( i < objMap_.size() );
	i = objMap_[i];
	assert( i < funcs_.size() );
	return i;
}

void Stoich::installReaction( ZeroOrder* forward, ZeroOrder* reverse, Id reacId )
{
	unsigned int rateIndex = convertIdToReacIndex( reacId );
	unsigned int revRateIndex = rateIndex;
	if ( useOneWay_ ) {
		rates_[ rateIndex ] = forward;
		revRateIndex = rateIndex + 1;
		rates_[ revRateIndex ] = reverse;
	} else {
		rates_[ rateIndex ] = 
			new BidirectionalReaction( forward, reverse );
	}

	vector< unsigned int > molIndex;

	if ( useOneWay_ ) {
		unsigned int numReactants = forward->getReactants( molIndex );
		for ( unsigned int i = 0; i < numReactants; ++i ) {
			int temp = N_.get( molIndex[i], rateIndex );
			N_.set( molIndex[i], rateIndex, temp - 1 );
			temp = N_.get( molIndex[i], revRateIndex );
			N_.set( molIndex[i], revRateIndex, temp + 1 );
		}

		numReactants = reverse->getReactants( molIndex );
		for ( unsigned int i = 0; i < numReactants; ++i ) {
			int temp = N_.get( molIndex[i], rateIndex );
			N_.set( molIndex[i], rateIndex, temp + 1 );
			temp = N_.get( molIndex[i], revRateIndex );
			N_.set( molIndex[i], revRateIndex, temp - 1 );
		}
	} else {
		unsigned int numReactants = forward->getReactants( molIndex );
		for ( unsigned int i = 0; i < numReactants; ++i ) {
			int temp = N_.get( molIndex[i], rateIndex );
			N_.set( molIndex[i], rateIndex, temp - 1 );
		}

		numReactants = reverse->getReactants( molIndex );
		for ( unsigned int i = 0; i < numReactants; ++i ) {
			int temp = N_.get( molIndex[i], revRateIndex );
			N_.set( molIndex[i], rateIndex, temp + 1 );
		}
	}
}

void Stoich::installMMenz( MMEnzymeBase* meb, unsigned int rateIndex,
	const vector< Id >& subs, const vector< Id >& prds )
{
	rates_[rateIndex] = meb;

	for ( unsigned int i = 0; i < subs.size(); ++i ) {
		unsigned int poolIndex = convertIdToPoolIndex( subs[i] );
		int temp = N_.get( poolIndex, rateIndex );
		N_.set( poolIndex, rateIndex, temp - 1 );
	}
	for ( unsigned int i = 0; i < prds.size(); ++i ) {
		unsigned int poolIndex = convertIdToPoolIndex( prds[i] );
		int temp = N_.get( poolIndex, rateIndex );
		N_.set( poolIndex, rateIndex, temp + 1 );
	}
}

void Stoich::installEnzyme( ZeroOrder* r1, ZeroOrder* r2, ZeroOrder* r3,
	Id enzId, Id enzMolId, const vector< Id >& prds ) 
{
	unsigned int rateIndex = convertIdToReacIndex( enzId );

	if ( useOneWay_ ) {
		rates_[ rateIndex ] = r1;
		rates_[ rateIndex + 1 ] = r2;
		rates_[ rateIndex + 2 ] = r3;
	} else {
		rates_[ rateIndex ] = new BidirectionalReaction( r1, r2 );
		rates_[ rateIndex + 1 ] = r3;
	}

	vector< unsigned int > poolIndex;
	unsigned int numReactants = r2->getReactants( poolIndex );
	assert( numReactants == 1 ); // Should be cplx as the only product
	unsigned int cplxPool = poolIndex[0];

	if ( useOneWay_ ) {
		numReactants = r1->getReactants( poolIndex ); // Substrates
		for ( unsigned int i = 0; i < numReactants; ++i ) {
			int temp = N_.get( poolIndex[i], rateIndex ); // terms for r1
			N_.set( poolIndex[i], rateIndex, temp - 1 );
			temp = N_.get( poolIndex[i], rateIndex + 1 ); //terms for r2
			N_.set( poolIndex[i], rateIndex + 1, temp + 1 );
		}

		int temp = N_.get( cplxPool, rateIndex );	// term for r1
		N_.set( cplxPool, rateIndex, temp + 1 );
		temp = N_.get( cplxPool, rateIndex + 1 );	// term for r2
		N_.set( cplxPool, rateIndex + 1, temp -1 );
	} else { // Regular bidirectional reactions.
		numReactants = r1->getReactants( poolIndex ); // Substrates
		for ( unsigned int i = 0; i < numReactants; ++i ) {
			int temp = N_.get( poolIndex[i], rateIndex );
			N_.set( poolIndex[i], rateIndex, temp - 1 );
		}
		int temp = N_.get( cplxPool, rateIndex );
		N_.set( cplxPool, rateIndex, temp + 1 );
	}

	// Now assign reaction 3. The complex is the only substrate here.
	// Reac 3 is already unidirectional, so all we need to do to handle
	// one-way reactions is to get the index right.
	unsigned int reac3index = ( useOneWay_ ) ? rateIndex + 2 : rateIndex + 1;
	int temp = N_.get( cplxPool, reac3index );
	N_.set( cplxPool, reac3index, temp - 1 );

	// For the products, we go to the prd list directly.
	for ( unsigned int i = 0; i < prds.size(); ++i ) {
		unsigned int j = convertIdToPoolIndex( prds[i] );
		int temp = N_.get( j, reac3index );
		N_.set( j, reac3index, temp + 1 );
	}
	// Enz is also a product here.
	unsigned int enzPool = convertIdToPoolIndex( enzMolId );
	temp = N_.get( enzPool, reac3index );
	N_.set( enzPool, reac3index, temp + 1 );
}

//////////////////////////////////////////////////////////////
// Field interface functions
//////////////////////////////////////////////////////////////

/**
 * Sets the forward rate v (given in millimoloar concentration units)
 * for the specified reaction throughout the compartment in which the
 * reaction lives. Internally the stoich uses #/voxel units so this 
 * involves querying the volume subsystem about volumes for each
 * voxel, and scaling accordingly.
 * For now assume a uniform voxel volume and hence just convert on 
 * 0 meshIndex.
 */
void Stoich::setReacKf( const Eref& e, double v ) const
{
	static const SrcFinfo* toSub = dynamic_cast< const SrcFinfo* > (
		ZombieReac::initCinfo()->findFinfo( "toSub" ) );

	assert( toSub );
	double volScale = convertConcToNumRateUsingMesh( e, toSub, 0 );

	rates_[ convertIdToReacIndex( e.id() ) ]->setR1( v / volScale );
}

/**
 * For now assume a single rate term.
 */
void Stoich::setReacKb( const Eref& e, double v ) const
{
	static const SrcFinfo* toPrd = static_cast< const SrcFinfo* > (
		ZombieReac::initCinfo()->findFinfo( "toPrd" ) );

	assert( toPrd );
	double volScale = convertConcToNumRateUsingMesh( e, toPrd, 0 );


	if ( useOneWay_ )
		 rates_[ convertIdToReacIndex( e.id() ) + 1 ]->setR1( v / volScale);
	else
		 rates_[ convertIdToReacIndex( e.id() ) ]->setR2( v / volScale );
}

void Stoich::setMMenzKm( const Eref& e, double v ) const
{
	static const SrcFinfo* toSub = dynamic_cast< const SrcFinfo* > (
		ZombieMMenz::initCinfo()->findFinfo( "toSub" ) );
	// Identify MMenz rate term
	RateTerm* rt = rates_[ convertIdToReacIndex( e.id() ) ];
	MMEnzymeBase* enz = dynamic_cast< MMEnzymeBase* >( rt );
	assert( enz );
	// Identify MMenz Enzyme substrate. I would have preferred the parent,
	// but that gets messy.
	// unsigned int enzMolIndex = enz->getEnzIndex();

	// This function can be replicated to handle multiple different voxels.
	vector< double > vols;
	getReactantVols( e, toSub, vols );
	if ( vols.size() == 0 ) {
		cerr << "Error: Stoich::setMMenzKm: no substrates for enzyme " <<
			e << endl;
		return;
	}
	// Do scaling and assignment.
	enz->setR1( v * vols[0] * NA );
}

void Stoich::setMMenzKcat( const Eref& e, double v ) const
{
	RateTerm* rt = rates_[ convertIdToReacIndex( e.id() ) ];
	MMEnzymeBase* enz = dynamic_cast< MMEnzymeBase* >( rt );
	assert( enz );

	enz->setR2( v );
}

/// Later handle all the volumes when this conversion is done.
void Stoich::setEnzK1( const Eref& e, double v ) const
{
	static const SrcFinfo* toSub = dynamic_cast< const SrcFinfo* > (
		ZombieEnz::initCinfo()->findFinfo( "toSub" ) );
	assert( toSub );

	double volScale = convertConcToNumRateUsingMesh( e, toSub, 1 );

	rates_[ convertIdToReacIndex( e.id() ) ]->setR1( v / volScale );
}

void Stoich::setEnzK2( const Eref& e, double v ) const
{
	if ( useOneWay_ )
		rates_[ convertIdToReacIndex( e.id() ) + 1 ]->setR1( v );
	else
		rates_[ convertIdToReacIndex( e.id() ) ]->setR2( v );
}

void Stoich::setEnzK3( const Eref& e, double v ) const
{
	if ( useOneWay_ )
		rates_[ convertIdToReacIndex( e.id() ) + 2 ]->setR1( v );
	else
		rates_[ convertIdToReacIndex( e.id() ) + 1 ]->setR1( v );
}

/**
 * Looks up the matching rate for R1. Later we may have additional 
 * scaling terms for the specified voxel.
 */
double Stoich::getR1( unsigned int reacIndex, unsigned int voxel ) const
{
	return rates_[ reacIndex ]->getR1();
}

/**
 * Looks up the matching rate for R2. Later we may have additional 
 * scaling terms for the specified voxel.
 */
double Stoich::getR2( unsigned int reacIndex, unsigned int voxel ) const
{
	return rates_[ reacIndex ]->getR2();
}

void Stoich::innerSetN( unsigned int meshIndex, Id id, double v )
{
	unsigned int poolIndex = convertIdToPoolIndex( id );
	assert( poolIndex < S_[meshIndex].size() );
	S_[ meshIndex ][ poolIndex ] = v;
	if ( poolIndex < numVarPools_ )
		y_[ meshIndex ][ poolIndex ] = v;
}

void Stoich::innerSetNinit( unsigned int meshIndex, Id id, double v )
{
	Sinit_[ meshIndex ][ convertIdToPoolIndex( id ) ] = v;
}

//////////////////////////////////////////////////////////////
// Model running functions
//////////////////////////////////////////////////////////////

// Update the v_ vector for individual reac velocities.
void Stoich::updateV( unsigned int meshIndex, vector< double >& v )
{
	// Some algorithm to assign the values from the computed rates
	// to the corresponding v_ vector entry
	// for_each( rates_.begin(), rates_.end(), assign);

	vector< RateTerm* >::const_iterator i;
	vector< double >::iterator j = v.begin();
	const double* S = &S_[meshIndex][0];

	for ( i = rates_.begin(); i != rates_.end(); i++)
	{
		*j++ = (**i)( S );
		assert( !isnan( *( j-1 ) ) );
	}

	// I should use forall here.
	/*
	vector< SumTotal >::const_iterator k;
	for ( k = sumTotals_.begin(); k != sumTotals_.end(); k++ )
		k->sum();
	*/
}

void Stoich::updateRates( vector< double>* yprime, double dt, 
	unsigned int meshIndex, vector< double >& v )
{
	updateV( meshIndex, v );

	// Much scope for optimization here.
	vector< double >::iterator j = yprime->begin();
	assert( yprime->size() >= numVarPools_ );
	for (unsigned int i = 0; i < numVarPools_; i++) {
		*j++ = dt * N_.computeRowRate( i , v );
	}
}

// Update the function-computed molecule terms. These are not integrated,
// but their values may be used by molecules that are.
// The molecule vector S_ has a section for FuncTerms. In this section
// there is a one-to-one match between entries in S_ and FuncTerm entries.
void Stoich::updateFuncs( double t, unsigned int meshIndex )
{
	vector< FuncTerm* >::const_iterator i;
	vector< double >::iterator j = S_[meshIndex].begin() + numVarPools_ + numBufPools_;

	for ( i = funcs_.begin(); i != funcs_.end(); i++)
	{
		*j++ = (**i)( &( S_[meshIndex][0] ), t );
		assert( !isnan( *( j-1 ) ) );
	}
}

/////////////////////////////////////////////////////////////////
// 
/////////////////////////////////////////////////////////////////
void Stoich::updateDiffusion( 
	unsigned int meshIndex, const vector< const Stencil* >& stencil)
{
	for ( unsigned int i = 0; i < stencil.size(); ++i ) {
		stencil[i]->addFlux( meshIndex, flux_[meshIndex], S_, diffConst_ );
	}
}

// void Stoich::clearFlux( unsigned int meshIndex )
// This variant also sends out the data for the current node.
// Problem here: I need to restrict myself to this thread's meshindices.
// Or pick the meshIndex == 0 and on that thread alone send out all the
// stuff. At this phase it should be safe as all the calculations are done,
// so none of the S_ vectors is changing.
void Stoich::clearFlux( unsigned int meshIndex, unsigned int threadNum )
{
	if ( meshIndex == 0 ) {
		for( vector< unsigned int >::iterator i = diffNodes_.begin(); 
			i != diffNodes_.end(); ++i )
		{
			vector< double > buf;
			for ( unsigned int j = 0; j < outgoing_.size(); ++j ) {
				buf.insert( buf.end(), S_[ outgoing_[*i][j] ].begin(),
					S_[ outgoing_[*i][j] ].end() );
			}
			nodeDiffBoundary()->send( stoichId_.eref(), threadNum, 
				*i, outgoing_[*i], buf );
		}
	}

	vector< double >& f = flux_[meshIndex];

	for ( vector< double >::iterator j = f.begin(); j != f.end(); ++j)
		*j = 0.0;
}

//// This belongs up in the destMsgs, but I'll put it here to help coding.
// It is the recipient function for the message sent at the node boundary.
// It can handle messages even if they are not destined for this node, it
// just ignores them.

void Stoich::handleNodeDiffBoundary( unsigned int nodeNum, 
	vector< unsigned int > meshEntries, vector< double > remoteS )
{
	vector< double >::iterator begin = remoteS.begin();
	for ( unsigned int i = 0; i < incoming_[ nodeNum ].size(); ++i ) {
		unsigned int meshIndex = incoming_[nodeNum][i];
		/// Check the syntax of assign.
		S_[meshIndex].assign( begin, begin + numVarPools_ );
		begin += numVarPools_;
	}
}

void Stoich::clearFlux()
{
	for ( unsigned int i = 0; i < flux_.size(); ++i )
		clearFlux( i, ScriptThreadNum );
}

// Put in a similar updateVals() function to handle Math expressions.
// Might update molecules, possibly even reac rates at some point.

/**
 * Assigns n values for all molecules that have had their slave-enable
 * flag set _after_ the run started. Ugly hack, but convenient for 
 * simulations. Likely to be very few, if any.
void Stoich::updateDynamicBuffers()
{
	// Here we handle dynamic buffering by simply writing over S_.
	// We never see y_ in the rest of the simulation, so can ignore.
	// Main concern is that y_ will go wandering off into nans, or
	// become numerically unhappy and slow things down.
	for ( vector< unsigned int >::const_iterator 
		i = dynamicBuffers_.begin(); i != dynamicBuffers_.end(); ++i )
		S_[ *i ] = Sinit_[ *i ];
}
 */

unsigned int Stoich::numMeshEntries() const
{
	return concInit_.size();
}

const double* Stoich::S( unsigned int meshIndex ) const
{
	return &S_[meshIndex][0];
}

double* Stoich::varS( unsigned int meshIndex )
{
	return &S_[meshIndex][0];
}

const double* Stoich::Sinit( unsigned int meshIndex ) const
{
	return &Sinit_[meshIndex][0];
}

void Stoich::setConcInit( unsigned int poolIndex, double conc )
{
	assert( conc >= 0 );
	assert( poolIndex < concInit_.size() );
	concInit_[poolIndex] = conc;
}

double Stoich::getDiffConst( unsigned int p ) const
{
	assert( p < diffConst_.size() );
	return diffConst_[p];
}

void Stoich::setDiffConst( unsigned int p, double d )
{
	assert( p < diffConst_.size() );
	if ( d < 0 ) {
		cout << "Warning: Stoich::setDiffConst: D[" << p << 
			"] cannot be -ve: " << d << endl;
		return;
	}
	diffConst_[p] = d;
}

SpeciesId Stoich::getSpecies( unsigned int poolIndex ) const
{
	return species_[ poolIndex ];
}

void Stoich::setSpecies( unsigned int poolIndex, SpeciesId s )
{
	species_[ poolIndex ] = s;
}

double* Stoich::getY( unsigned int meshIndex )
{
	return &y_[meshIndex][0];
}

void Stoich::print() const
{
	N_.print();
}

#ifdef USE_GSL
///////////////////////////////////////////////////
// GSL interface stuff
///////////////////////////////////////////////////

/**
 * This is the function used by GSL to advance the simulation one step.
 * We have a design decision here: to perform the calculations 'in place'
 * on the passed in y and f arrays, or to copy the data over and use
 * the native calculations in the Stoich object. We chose the latter,
 * because memcpy is fast, and the alternative would be to do a huge
 * number of array lookups (currently it is direct pointer references).
 * Someday should benchmark to see how well it works.
 * The derivative array f is used directly by the stoich function
 * updateRates that computes these derivatives, so we do not need to
 * do any memcopies there.
 *
 * Perhaps not by accident, this same functional form is used by CVODE.
 * Should make it easier to eventually use CVODE as a solver too.
 */

// Static function passed in as the stepper for GSL
int Stoich::gslFunc( double t, const double* y, double* yprime, void* s )
{
	StoichThread* st = static_cast< StoichThread* >( s );
	return st->stoich()->innerGslFunc( t, y, yprime, st->meshIndex() );
	// return static_cast< Stoich* >( s )->innerGslFunc( t, y, yprime );
}


int Stoich::innerGslFunc( double t, const double* y, double* yprime, 
	unsigned int meshIndex )
{
	vector< double > v( numReac_ );
	// Copy the y array into the S_ vector.
	// Sometimes GSL passes in its own allocated version of y.
	/*
	unsigned int begin = ( numMeshEntries_ * p->threadIndexInGroup ) /
		p->numThreadsInGroup;
	unsigned int end = ( numMeshEntries_ * ( 1 + p->threadIndexInGroup ) ) /
		p->numThreadsInGroup;
		*/
	memcpy( &S_[meshIndex][0], y, numVarPoolsBytes_ );

	//	updateDynamicBuffers();
	updateFuncs( t, meshIndex );

	updateV( meshIndex, v );

	// updateDiffusion happens in the previous Process Tick, coordinated
	// by the MeshEntries. At this point the new values are there in the
	// flux_ matrix.

	// Much scope for optimization here.
	const vector< double >& f = flux_[ meshIndex ];
	for (unsigned int i = 0; i < numVarPools_; i++) {
		*yprime++ = N_.computeRowRate( i , v ) + f[i];
	}
	// cout << t << ": " << y[0] << ", " << y[1] << endl;
	return GSL_SUCCESS;
}

#endif // USE_GSL
