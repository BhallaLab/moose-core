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
#include "PoolBase.h"
#include "Pool.h"
#include "BufPool.h"
#include "FuncPool.h"
#include "ReacBase.h"
#include "Reac.h"
#include "EnzBase.h"
#include "CplxEnzBase.h"
#include "Enz.h"
#include "MMenz.h"
#include "SumFunc.h"
#include "MathFunc.h"
#include "StoichPools.h"
#include "ZPool.h"
#include "ZombiePool.h"
#include "ZombieBufPool.h"
#include "ZombieFuncPool.h"
#include "ZReac.h"
#include "ZombieEnz.h"
#include "ZombieMMenz.h"
#include "ZombieSumFunc.h"
#include "../shell/Shell.h"

#ifdef USE_GSL
#include <gsl/gsl_errno.h>
#endif

#define EPSILON 1e-15

const Cinfo* StoichCore::initCinfo()
{
		//////////////////////////////////////////////////////////////
		// Field Definitions
		//////////////////////////////////////////////////////////////
		static ValueFinfo< StoichCore, bool > useOneWay(
			"useOneWayReacs",
			"Flag: use bidirectional or one-way reacs. One-way is needed"
			"for Gillespie type stochastic calculations. Two-way is"
			"likely to be margninally more efficient in ODE calculations",
			&StoichCore::setOneWay,
			&StoichCore::getOneWay
		);

		static ReadOnlyValueFinfo< StoichCore, unsigned int > nVarPools(
			"nVarPools",
			"Number of variable molecule pools in the reac system",
			&StoichCore::getNumVarPools
		);

		static ElementValueFinfo< StoichCore, string > path(
			"path",
			"Path of reaction system to take over",
			&StoichCore::setPath,
			&StoichCore::getPath
		);

		static ReadOnlyValueFinfo< StoichCore, double > estimatedDt(
			"estimatedDt",
			"Estimate of fastest (smallest) timescale in system."
			"This is fallible because it depends on instantaneous concs,"
			"which of course change over the course of the simulation.",
			&StoichCore::getEstimatedDt
		);

	static Finfo* stoichCoreFinfos[] = {
		&useOneWay,		// Value
		&nVarPools,		// Value
		&estimatedDt,		// ReadOnlyValue
		&path,			// Value
	};

	static Cinfo stoichCoreCinfo (
		"StoichCore",
		Neutral::initCinfo(),
		stoichCoreFinfos,
		sizeof( stoichCoreFinfos ) / sizeof ( Finfo* ),
		new Dinfo< StoichCore >()
	);

	return &stoichCoreCinfo;
}

//////////////////////////////////////////////////////////////
// Class definitions
//////////////////////////////////////////////////////////////
static const Cinfo* stoichCoreCinfo = StoichCore::initCinfo();

StoichCore::StoichCore()
	: 
		useOneWay_( 0 ),
		path_( "" ),
		objMapStart_( 0 ),
		numVarPools_( 0 ),
		numVarPoolsBytes_( 0 ),
		numBufPools_( 0 ),
		numFuncPools_( 0 ),
		numReac_( 0 )
{;}

StoichCore::~StoichCore()
{
	unZombifyModel();
	for ( vector< RateTerm* >::iterator i = rates_.begin();
		i != rates_.end(); ++i )
		delete *i;

	for ( vector< FuncTerm* >::iterator i = funcs_.begin();
		i != funcs_.end(); ++i )
		delete *i;
}

//////////////////////////////////////////////////////////////
// Field Definitions
//////////////////////////////////////////////////////////////

void StoichCore::setOneWay( bool v )
{
	useOneWay_ = v;
}

bool StoichCore::getOneWay() const
{
	return useOneWay_;
}

void StoichCore::setPath( const Eref& e, const Qinfo* q, string v )
{
	if ( path_ != "" && path_ != v ) {
		// unzombify( path_ );
		cout << "StoichCore::setPath: need to clear old path.\n";
		return;
	}
	path_ = v;
	vector< Id > elist;
	Shell::wildcard( path_, elist );

	allocateObjMap( elist );
	allocateModel( elist );
	ObjId pa = Neutral::parent( e );
	if ( pa.element()->cinfo()->findFinfo( "stoich" ) )
		SetGet1< Id >::setRepeat( pa.id, "stoich", e.id() );
	zombifyModel( e, elist );
}

string StoichCore::getPath( const Eref& e, const Qinfo* q ) const
{
	return path_;
}

double StoichCore::getEstimatedDt() const
{
	return 1; // Dummy
}

unsigned int StoichCore::getNumVarPools() const
{
	return numVarPools_;
}

unsigned int StoichCore::getNumAllPools() const
{
	assert( diffConst_.size() == 
					numVarPools_ + numBufPools_ + numFuncPools_ );
	return numVarPools_ + numBufPools_ + numFuncPools_;
}

//////////////////////////////////////////////////////////////
// Model zombification functions
//////////////////////////////////////////////////////////////
void StoichCore::allocateObjMap( const vector< Id >& elist )
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

/// Identifies and allocates objects in the StoichCore.
void StoichCore::allocateModelObject( Id id, 
				vector< Id >& bufPools, vector< Id >& funcPools )
{
	static const Cinfo* poolCinfo = Pool::initCinfo();
	static const Cinfo* bufPoolCinfo = BufPool::initCinfo();
	static const Cinfo* funcPoolCinfo = FuncPool::initCinfo();
	static const Cinfo* reacCinfo = Reac::initCinfo();
	static const Cinfo* enzCinfo = Enz::initCinfo();
	static const Cinfo* mmEnzCinfo = MMenz::initCinfo();
	static const Cinfo* sumFuncCinfo = SumFunc::initCinfo();

	Element* ei = id.element();
	if ( ei->cinfo() == poolCinfo ) {
		objMap_[ id.value() - objMapStart_ ] = numVarPools_;
		idMap_.push_back( id );
		++numVarPools_;
	} else if ( ei->cinfo() == bufPoolCinfo ) {
			bufPools.push_back( id );
	} else if ( ei->cinfo() == funcPoolCinfo ) {
			funcPools.push_back( id );
	} else if ( ei->cinfo() == mmEnzCinfo ){
			mmEnzMap_.push_back( ei->id() );
			objMap_[ id.value() - objMapStart_ ] = numReac_;
			++numReac_;
	} else if ( ei->cinfo() == reacCinfo ) {
			reacMap_.push_back( ei->id() );
			if ( useOneWay_ ) {
				objMap_[ id.value() - objMapStart_ ] = numReac_;
				numReac_ += 2;
			} else {
				objMap_[ id.value() - objMapStart_ ] = numReac_;
				++numReac_;
			}
	} else if ( ei->cinfo() == enzCinfo ) {
			enzMap_.push_back( ei->id() );
			if ( useOneWay_ ) {
				objMap_[ id.value() - objMapStart_ ] = numReac_;
				numReac_ += 3;
			} else {
				objMap_[ id.value() - objMapStart_ ] = numReac_;
				numReac_ += 2;
			}
	} else if ( ei->cinfo() == sumFuncCinfo ){
			objMap_[ id.value() - objMapStart_ ] = numFuncPools_;
			++numFuncPools_;
	} 
}

/// Using the computed array sizes, now allocate space for them.
void StoichCore::resizeArrays()
{
	/*
	S_.resize( 1 );
	Sinit_.resize( 1 );
	S_[0].resize( numVarPools_ + numBufPools_ + numFuncPools_, 0.0 );
	Sinit_[0].resize( numVarPools_ + numBufPools_ + numFuncPools_, 0.0);
	*/

	diffConst_.resize( numVarPools_ + numBufPools_ + numFuncPools_, 0.0 );
	species_.resize( numVarPools_ + numBufPools_ + numFuncPools_, 0 );
	rates_.resize( numReac_ );
	// v_.resize( numReac_, 0.0 ); // v is now allocated dynamically
	funcs_.resize( numFuncPools_ );
	N_.setSize( numVarPools_ + numBufPools_ + numFuncPools_, numReac_ );
}

/// Calculate sizes of all arrays, and allocate them.
void StoichCore::allocateModel( const vector< Id >& elist )
{
	numVarPools_ = 0;
	numReac_ = 0;
	numFuncPools_ = 0;
	vector< Id > bufPools;
	vector< Id > funcPools;
	idMap_.clear();
	reacMap_.clear();
	enzMap_.clear();
	mmEnzMap_.clear();

	for ( vector< Id >::const_iterator i = elist.begin(); 
					i != elist.end(); ++i )
			allocateModelObject( *i, bufPools, funcPools );

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
	numVarPoolsBytes_ = numVarPools_ * sizeof( double );

	resizeArrays();
}

static void zombifyAndUnschedPool( 
	const Eref& s, Element* orig, const Cinfo* zClass )
{
	////////////////////////////////////////////////////////
	// Unschedule: Get rid of Process message
	static const Finfo* procDest = 
		PoolBase::initCinfo()->findFinfo( "process");
	assert( procDest );

	const DestFinfo* df = dynamic_cast< const DestFinfo* >( procDest );
	assert( df );
	MsgId mid = orig->findCaller( df->getFid() );
	if ( mid != Msg::bad )
		Msg::deleteMsg( mid );

	// Complete the unzombification.
	PoolBase::zombify( orig, zClass, s.id() );
}

// e is the stoich Eref, elist is list of all Ids to zombify.
void StoichCore::zombifyModel( const Eref& e, const vector< Id >& elist )
{
	static const Cinfo* poolCinfo = Pool::initCinfo();
	static const Cinfo* bufPoolCinfo = BufPool::initCinfo();
	static const Cinfo* funcPoolCinfo = FuncPool::initCinfo();
	static const Cinfo* reacCinfo = Reac::initCinfo();
	static const Cinfo* enzCinfo = Enz::initCinfo();
	static const Cinfo* mmEnzCinfo = MMenz::initCinfo();
	vector< Id > meshEntries;

	ObjId stoichParent = Neutral::parent( e );
	assert( stoichParent.element()->cinfo()->isA( "GslStoich" ) ||
		stoichParent.element()->cinfo()->isA( "GssaStoich" ) );

	for ( vector< Id >::const_iterator i = elist.begin(); i != elist.end(); ++i ){
		Element* ei = (*i)();
		if ( ei->cinfo() == poolCinfo ) {
			zombifyAndUnschedPool( stoichParent.eref(), 
							(*i)(), ZPool::initCinfo() );
		}
		else if ( ei->cinfo() == bufPoolCinfo ) {
			zombifyAndUnschedPool( e, (*i)(), ZombieBufPool::initCinfo() );
		}
		else if ( ei->cinfo() == funcPoolCinfo ) {
			zombifyAndUnschedPool( e, (*i)(), ZombieFuncPool::initCinfo());
			// Has also got to zombify the Func.
			Id funcId = Neutral::child( i->eref(), "sumFunc" );
			if ( funcId != Id() ) {
				if ( funcId()->cinfo()->isA( "SumFunc" ) )
					ZombieSumFunc::zombify( e.element(), funcId(), (*i) );
			}
		}
		else if ( ei->cinfo() == reacCinfo ) {
			ReacBase::zombify( ei, ZReac::initCinfo(), e.id() );
		}
		else if ( ei->cinfo() == mmEnzCinfo ) {
			EnzBase::zombify( ei, ZombieMMenz::initCinfo(), e.id() );
		}
		else if ( ei->cinfo() == enzCinfo ) {
			CplxEnzBase::zombify( ei, ZombieEnz::initCinfo(), e.id() );
		}
	}
}

void StoichCore::unZombifyPools()
{
	unsigned int i = 0;
	for ( ; i < numVarPools_; ++i ) {
		Element* e = idMap_[i].element();
		if ( e != 0 &&  e->cinfo() == ZPool::initCinfo() )
			PoolBase::zombify( e, Pool::initCinfo(), Id() );
	}
	
	for ( ; i < numVarPools_ + numBufPools_; ++i ) {
		Element* e = idMap_[i].element();
		if ( e != 0 &&  e->cinfo() == ZombieBufPool::initCinfo() )
			PoolBase::zombify( e, BufPool::initCinfo(), Id() );
	}
}

void StoichCore::unZombifyFuncs()
{
	unsigned int start = numVarPools_ + numBufPools_;
	for ( unsigned int k = 0; k < numFuncPools_; ++k ) {
		unsigned int i = k + start;
		Element* e = idMap_[i].element();
		if ( e != 0 &&  e->cinfo() == ZombieFuncPool::initCinfo() ) {
			PoolBase::zombify( e, FuncPool::initCinfo(), Id() );
			// Has also got to unzombify the Func.
			Id funcId = Neutral::child( idMap_[i].eref(), "sumFunc" );
			if ( funcId != Id() ) {
				assert ( funcId()->cinfo()->isA( "ZombieSumFunc" ) );
				ZombieSumFunc::unzombify( funcId.element() );
			}
		}
	}
}

void StoichCore::unZombifyModel()
{
	assert (idMap_.size() == numVarPools_ + numBufPools_ + numFuncPools_);

	unZombifyPools();
	unZombifyFuncs();

	Shell* s = reinterpret_cast< Shell* >( Id().eref().data() );

	for ( vector< Id >::iterator i = reacMap_.begin(); 
						i != reacMap_.end(); ++i ) {
		Element* e = i->element();
		if ( e != 0 &&  e->cinfo() == ZReac::initCinfo() )
			ReacBase::zombify( e, Reac::initCinfo(), Id() );
	}
	
	for ( vector< Id >::iterator i = mmEnzMap_.begin(); 
						i != mmEnzMap_.end(); ++i ) {
		Element* e = i->element();
		if ( e != 0 &&  e->cinfo() == ZombieMMenz::initCinfo() )
			EnzBase::zombify( e, MMenz::initCinfo(), Id() );
	}
	
	for ( vector< Id >::iterator i = enzMap_.begin(); 
						i != enzMap_.end(); ++i ) {
		Element* e = i->element();
		if ( e != 0 &&  e->cinfo() == ZombieEnz::initCinfo() )
			CplxEnzBase::zombify( e, Enz::initCinfo(), Id() );
	}

	s->addClockMsgs( idMap_, "proc", 4 );
}

unsigned int StoichCore::convertIdToPoolIndex( Id id ) const
{
	unsigned int i = id.value() - objMapStart_;
	assert( i < objMap_.size() );
	i = objMap_[i];
	assert( i < numVarPools_ + numBufPools_ + numFuncPools_ );
	return i;
}

unsigned int StoichCore::convertIdToReacIndex( Id id ) const
{
	unsigned int i = id.value() - objMapStart_;
	assert( i < objMap_.size() );
	i = objMap_[i];
	assert( i < rates_.size() );
	return i;
}

unsigned int StoichCore::convertIdToFuncIndex( Id id ) const
{
	unsigned int i = id.value() - objMapStart_;
	assert( i < objMap_.size() );
	i = objMap_[i];
	assert( i < funcs_.size() );
	return i;
}

void StoichCore::installReaction( ZeroOrder* forward, ZeroOrder* reverse, Id reacId )
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

void StoichCore::installMMenz( MMEnzymeBase* meb, unsigned int rateIndex,
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

void StoichCore::installEnzyme( ZeroOrder* r1, ZeroOrder* r2, ZeroOrder* r3,
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
void StoichCore::setReacKf( const Eref& e, double v ) const
{
	static const SrcFinfo* toSub = dynamic_cast< const SrcFinfo* > (
		ZReac::initCinfo()->findFinfo( "toSub" ) );

	assert( toSub );
	double volScale = convertConcToNumRateUsingMesh( e, toSub, 0 );

	rates_[ convertIdToReacIndex( e.id() ) ]->setR1( v / volScale );
}

/**
 * For now assume a single rate term.
 */
void StoichCore::setReacKb( const Eref& e, double v ) const
{
	static const SrcFinfo* toPrd = static_cast< const SrcFinfo* > (
		ZReac::initCinfo()->findFinfo( "toPrd" ) );

	assert( toPrd );
	double volScale = convertConcToNumRateUsingMesh( e, toPrd, 0 );


	if ( useOneWay_ )
		 rates_[ convertIdToReacIndex( e.id() ) + 1 ]->setR1( v / volScale);
	else
		 rates_[ convertIdToReacIndex( e.id() ) ]->setR2( v / volScale );
}

void StoichCore::setMMenzKm( const Eref& e, double v ) const
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
		cerr << "Error: StoichCore::setMMenzKm: no substrates for enzyme " <<
			e << endl;
		return;
	}
	// Do scaling and assignment.
	enz->setR1( v * vols[0] * NA );
}

void StoichCore::setMMenzKcat( const Eref& e, double v ) const
{
	RateTerm* rt = rates_[ convertIdToReacIndex( e.id() ) ];
	MMEnzymeBase* enz = dynamic_cast< MMEnzymeBase* >( rt );
	assert( enz );

	enz->setR2( v );
}

/// Later handle all the volumes when this conversion is done.
void StoichCore::setEnzK1( const Eref& e, double v ) const
{
	static const SrcFinfo* toSub = dynamic_cast< const SrcFinfo* > (
		ZombieEnz::initCinfo()->findFinfo( "toSub" ) );
	assert( toSub );

	double volScale = convertConcToNumRateUsingMesh( e, toSub, 1 );

	rates_[ convertIdToReacIndex( e.id() ) ]->setR1( v / volScale );
}

void StoichCore::setEnzK2( const Eref& e, double v ) const
{
	if ( useOneWay_ )
		rates_[ convertIdToReacIndex( e.id() ) + 1 ]->setR1( v );
	else
		rates_[ convertIdToReacIndex( e.id() ) ]->setR2( v );
}

void StoichCore::setEnzK3( const Eref& e, double v ) const
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
double StoichCore::getR1( unsigned int reacIndex, unsigned int voxel ) const
{
	return rates_[ reacIndex ]->getR1();
}

/**
 * Looks up the matching rate for R2. Later we may have additional 
 * scaling terms for the specified voxel.
 */
double StoichCore::getR2( unsigned int reacIndex, unsigned int voxel ) const
{
	return rates_[ reacIndex ]->getR2();
}

double StoichCore::getDiffConst( unsigned int p ) const
{
	assert( p < diffConst_.size() );
	return diffConst_[p];
}

void StoichCore::setDiffConst( unsigned int p, double d )
{
	assert( p < diffConst_.size() );
	if ( d < 0 ) {
		cout << "Warning: StoichCore::setDiffConst: D[" << p << 
			"] cannot be -ve: " << d << endl;
		return;
	}
	diffConst_[p] = d;
}

SpeciesId StoichCore::getSpecies( unsigned int poolIndex ) const
{
	return species_[ poolIndex ];
}

void StoichCore::setSpecies( unsigned int poolIndex, SpeciesId s )
{
	species_[ poolIndex ] = s;
}

// for debugging.
void StoichCore::print() const
{
	N_.print();
}


/////////////////////////////////////////////////////////////////////
// Numeric funcs. These are in StoichCore because the rate terms are here.
/////////////////////////////////////////////////////////////////////

/**
 * updateRates computes the velocity *v* of each reaction. Then it
 * uses this to compute the rate of change, *yprime*, for each pool
 */

void StoichCore::updateRates( const double* s, double* yprime )
{
	vector< RateTerm* >::const_iterator i;
	vector< double > v( numReac_ );
	vector< double >::iterator j = v.begin();

	for ( i = rates_.begin(); i != rates_.end(); i++) {
		*j++ = (**i)( s );
		assert( !isnan( *( j-1 ) ) );
	}

	for (unsigned int i = 0; i < numVarPools_; ++i)
		*yprime++ = N_.computeRowRate( i , v );
}

/*
void StoichCore::updateRates( double* yprime, const vector< double >&v )
{
	for (unsigned int i = 0; i < numVarPools_; ++i)
		*yprime++ = N_.computeRowRate( i , v );
}

void StoichCore::updateV( const double* s, vector< double >& v )
{
	vector< RateTerm* >::const_iterator i;
	vector< double >::iterator j = v.begin();

	for ( i = rates_.begin(); i != rates_.end(); i++) {
		*j++ = (**i)( s );
		assert( !isnan( *( j-1 ) ) );
	}
}
*/

// s is the array of pools, S_[meshIndex][0]
void StoichCore::updateFuncs( double* s, double t )
{
	double* j = s + numVarPools_ + numBufPools_;

	for ( vector< FuncTerm* >::iterator i = funcs_.begin();
					i != funcs_.end(); ++i ) {
		*j++ = (**i)( s, t );
		assert( !isnan( *(j-1) ) );
	}
}
