/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
#include "ElementValueFinfo.h"
#include "PoolBase.h"
#include "ReacBase.h"
#include "EnzBase.h"
#include "CplxEnzBase.h"
#include "RateTerm.h"
#include "FuncTerm.h"
#include "SumTotalTerm.h"
#include "FuncBase.h"
#include "SparseMatrix.h"
#include "KinSparseMatrix.h"
#include "Stoich.h"
#include "lookupVolumeFromMesh.h"
#include "../shell/Shell.h"
#include "../shell/Wildcard.h"

#ifdef USE_GSL
#include <gsl/gsl_errno.h>
#endif

#define EPSILON 1e-15

const Cinfo* Stoich::initCinfo()
{
		//////////////////////////////////////////////////////////////
		// Field Definitions
		//////////////////////////////////////////////////////////////
		static ElementValueFinfo< Stoich, string > path(
			"path",
			"Wildcard path for reaction system handled by Stoich",
			&Stoich::setPath,
			&Stoich::getPath
		);

		static ValueFinfo< Stoich, Id > poolInterface(
			"poolInterface",
			"Accessory class that provides interface for accessing all "
		    " the pools that are present in this reaction system."
		    " Must be of class Ksolve or Dsolve (at present) "
			" Must be assigned before the path is set.",
			&Stoich::setPoolInterface,
			&Stoich::getPoolInterface
		);

		static ReadOnlyValueFinfo< Stoich, double > estimatedDt(
			"estimatedDt",
			"Estimated timestep for reac system based on Euler error",
			&Stoich::getEstimatedDt
		);

		static ReadOnlyValueFinfo< Stoich, unsigned int > numVarPools(
			"numVarPools",
			"Number of time-varying pools to be computed by the "
			"numerical engine",
			&Stoich::getNumVarPools
		);

		static ReadOnlyValueFinfo< Stoich, unsigned int > numAllPools(
			"numAllPools",
			"Total number of pools handled by the numerical engine. "
			"This includes variable ones, buffered ones, and functions",
			&Stoich::getNumAllPools
		);

		static ReadOnlyValueFinfo< Stoich, unsigned int > numRates(
			"numRates",
			"Total number of rate terms in the reaction system.",
			&Stoich::getNumRates
		);
		// Stuff here for getting Stoichiometry matrix to manipulate in
		// Python.

		//////////////////////////////////////////////////////////////
		// MsgDest Definitions
		//////////////////////////////////////////////////////////////

		//////////////////////////////////////////////////////////////
		// SrcFinfo Definitions
		//////////////////////////////////////////////////////////////

		//////////////////////////////////////////////////////////////
		// SharedMsg Definitions
		//////////////////////////////////////////////////////////////

	static Finfo* stoichFinfos[] = {
		&path,				// ElementValue
		&poolInterface,		// Value
		&estimatedDt,		// ReadOnlyValue
		&numVarPools,		// ReadOnlyValue
		&numAllPools,		// ReadOnlyValue
		&numRates,			// ReadOnlyValue
	};

	static Dinfo< Stoich > dinfo;
	static Cinfo stoichCinfo (
		"Stoich",
		Neutral::initCinfo(),
		stoichFinfos,
		sizeof( stoichFinfos ) / sizeof ( Finfo* ),
		&dinfo
	);

	return &stoichCinfo;
}

//////////////////////////////////////////////////////////////
// Class definitions
//////////////////////////////////////////////////////////////
static const Cinfo* stoichCinfo = Stoich::initCinfo();

//////////////////////////////////////////////////////////////

Stoich::Stoich()
	: 
		useOneWay_( 0 ),
		path_( "" ),
		poolInterface_(), // Must be reassigned to build stoich system.
		objMapStart_( 0 ),
		numVarPools_( 0 ),
		numVarPoolsBytes_( 0 ),
		numBufPools_( 0 ),
		numFuncPools_( 0 ),
		numReac_( 0 )
{;}

Stoich::~Stoich()
{
	unZombifyModel();
	for ( vector< RateTerm* >::iterator i = rates_.begin();
		i != rates_.end(); ++i )
		delete *i;

	/*
	 * Do NOT delete FuncTerms, they are just pointers stolen from
	 * the non-zombified objects.
	for ( vector< FuncTerm* >::iterator i = funcs_.begin();
		i != funcs_.end(); ++i )
		delete *i;
		*/
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

void Stoich::setPath( const Eref& e, string v )
{
	if ( path_ != "" && path_ != v ) {
		// unzombify( path_ );
		cout << "Stoich::setPath: need to clear old path.\n";
		return;
	}
	vector< ObjId > elist;
	path_ = v;
	wildcardFind( path_, elist );
	setElist( e, elist );
}

void convWildcards( vector< Id >& ret, const vector< ObjId >& elist )
{
	ret.resize( elist.size() );
	for ( unsigned int i = 0; i < elist.size(); ++i )
		ret[i] = elist[i].id;
}

void filterWildcards( vector< Id >& ret, const vector< ObjId >& elist )
{
	ret.clear();
	ret.reserve( elist.size() );
	for ( vector< ObjId >::const_iterator 
			i = elist.begin(); i != elist.end(); ++i ) {
		if ( i->element()->cinfo()->isA( "PoolBase" ) ||
			i->element()->cinfo()->isA( "ReacBase" ) ||
			i->element()->cinfo()->isA( "EnzBase" ) ||
			i->element()->cinfo()->isA( "FuncBase" ) )
		ret.push_back( i->id );
	}
}

void Stoich::setElist( const Eref& e, const vector< ObjId >& elist )
{
	path_ = "elist";
	ObjId myCompt = getCompt( e.id() );
	vector< Id > temp;
	filterWildcards( temp, elist );

	locateOffSolverReacs( myCompt, temp );
	allocateObjMap( temp );
	allocateModel( temp );
	zombifyModel( e, temp );
}

string Stoich::getPath( const Eref& e ) const
{
	return path_;
}

void Stoich::setPoolInterface( Id zpi ) {
	if ( ! ( 
			zpi.element()->cinfo()->isA( "Ksolve" )  ||
			zpi.element()->cinfo()->isA( "Dsolve" ) 
		   )
	   ) {
		cout << "Error: Stoich::setPoolInterface: invalid class assigned,"
				" should be either Ksolve or Dsolve\n";
		return;
	}
	poolInterface_ = zpi;
}

Id Stoich::getPoolInterface() const
{
	return poolInterface_;
}

double Stoich::getEstimatedDt() const
{
	return 1; // Dummy
}

unsigned int Stoich::getNumVarPools() const
{
	return numVarPools_;
}

unsigned int Stoich::getNumAllPools() const
{
	return numVarPools_ + numBufPools_ + numFuncPools_;
}

unsigned int Stoich::getNumProxyPools() const
{
	return offSolverPools_.size();
}

unsigned int Stoich::getNumRates() const
{
	return rates_.size();
}

const RateTerm* Stoich::rates( unsigned int i ) const
{
	assert( i < rates_.size() );
	return rates_[i];
}

//////////////////////////////////////////////////////////////
// Model zombification functions
//////////////////////////////////////////////////////////////

/**
 * Checks if specified reac is off solver. As side-effect it compiles
 * a vector of the pools that are off-solver, and the corresponding 
 * compartments for those pools
 */
static bool isOffSolverReac( const Element* e, Id myCompt, 
		vector< Id >& offSolverPools, vector< Id >& poolCompts,
		map< Id, Id >& poolComptMap )
{
	assert( myCompt != Id() );
	assert( myCompt.element()->cinfo()->isA( "ChemCompt" ) );
	bool ret = false;
	vector< Id > neighbours;
	e->getNeighbours( neighbours, e->cinfo()->findFinfo( "subOut" ));
	vector< Id > n2;
	e->getNeighbours( n2, e->cinfo()->findFinfo( "prdOut" ));
	neighbours.insert( neighbours.end(), n2.begin(), n2.end() );
	for ( vector< Id >::const_iterator 
			j = neighbours.begin(); j != neighbours.end(); ++j )
	{
		Id otherCompt = getCompt( *j );
		if ( myCompt != otherCompt ) {
			offSolverPools.push_back( *j );
			poolCompts.push_back( otherCompt );
			poolComptMap[ *j ] = otherCompt; // Avoids duplication of pools
			ret = true;
		}
	}
	return ret;
}

/**
 * Extracts and orders the compartments associated with a given reac.
 */
pair< Id, Id > extractCompts( const vector< Id >& compts )
{
	pair< Id, Id > ret;
	for ( vector< Id >::const_iterator i = compts.begin(); 
						i != compts.end(); ++i )
	{
		if ( ret.first == Id() ) {
			ret.first = *i;
		} else if ( ret.first != *i ) {
			if ( ret.second == Id() )
				ret.second = *i;
			else {
				cout << "Error: extractCompts: more than 2 compartments\n";
				assert( 0 );
			}
		}
	}
	if ( ( ret.second != Id() ) && ret.second < ret.first ) {
		Id temp = ret.first;
		ret.first = ret.second;
		ret.second = ret.first;
	}       

	return ret;
}

void Stoich::locateOffSolverReacs( Id myCompt, vector< Id >& elist )
{
	offSolverPools_.clear();
	offSolverReacs_.clear();
	offSolverReacCompts_.clear();
	map< Id, Id > poolComptMap; // < pool, compt >

	vector< Id > temp;
	temp.reserve( elist.size() );
	for ( vector< Id >::const_iterator 
					i = elist.begin(); i != elist.end(); ++i )
	{
		const Element* e = i->element();
		if ( e->cinfo()->isA( "ReacBase" ) || e->cinfo()->isA( "EnzBase" ) )
	   	{
			vector< Id > compts;
			if ( isOffSolverReac( e, myCompt, offSolverPools_, compts,
							   poolComptMap	) ) {
				offSolverReacs_.push_back( *i );
				offSolverReacCompts_.push_back( extractCompts( compts ) );
			} else  {
				temp.push_back( *i );
			}
		} else {
			temp.push_back( *i );
		}
	}

	offSolverPoolMap_.clear();
	for ( map< Id, Id >::iterator 
		i = poolComptMap.begin(); i != poolComptMap.end(); ++i ) {
		// fill in the map for activeOffSolverPools.
		offSolverPoolMap_[i->second].push_back( i->first );
	}

	// Ensure we don't have repeats, and the pools are ordered by compt
	offSolverPools_.clear();
	for ( map< Id, vector< Id > >::iterator
		i = offSolverPoolMap_.begin(); i != offSolverPoolMap_.end(); ++i ){
			if ( i->first != myCompt ) {
				offSolverPools_.insert( offSolverPools_.end(), 
					i->second.begin(), i->second.end() );
			}
	}

	elist = temp;
}

void Stoich::allocateObjMap( const vector< Id >& elist )
{
	vector< Id > temp( elist );
	temp.insert( temp.end(), offSolverPools_.begin(), 
					offSolverPools_.end() );
	temp.insert( temp.end(), offSolverReacs_.begin(), 
					offSolverReacs_.end() );
	objMapStart_ = ~0;
	unsigned int maxId = 0;
	for ( vector< Id >::const_iterator 
					i = temp.begin(); i != temp.end(); ++i ) {
		if ( objMapStart_ > i->value() )
			objMapStart_ = i->value();
		if ( maxId < i->value() )
			maxId = i->value();
	}
	objMap_.clear();
	objMap_.resize( 1 + maxId - objMapStart_, 0 );
	/**
	 * If this assertion fails it usually means that the elist passed to
	 * the solver is not properly restricted to objects located on the
	 * current compartment. As a result of this, traversal for finding
	 * off-compartment pools generates repeats with the ones in the elist.
	 * Note that pool compartment assignment is determined by following
	 * the mesh message, and thus a tree-based elist construction for
	 * compartments may be incompatible with the generation of the lists
	 * of off-compartment pools. It is up to the upstream code to
	 * ensure that this is done properly.
	 *
	 * This assertion also fails if the Ids concerned had a dimension
	 * greater than 1.
	 */
	assert( objMap_.size() >= temp.size() );
}

/// Identifies and allocates objects in the Stoich.
void Stoich::allocateModelObject( Id id, 
				vector< Id >& bufPools, vector< Id >& funcPools )
{
	static const Cinfo* poolCinfo = Cinfo::find( "Pool" );
	static const Cinfo* bufPoolCinfo = Cinfo::find( "BufPool" );
	static const Cinfo* funcPoolCinfo = Cinfo::find( "FuncPool" );
	static const Cinfo* reacCinfo = Cinfo::find( "Reac" );
	static const Cinfo* enzCinfo = Cinfo::find( "Enz" );
	static const Cinfo* mmEnzCinfo = Cinfo::find( "MMenz" );
	static const Cinfo* sumFuncCinfo = Cinfo::find( "SumFunc" );

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
void Stoich::resizeArrays()
{
	assert( idMap_.size() == numVarPools_ + numBufPools_ + numFuncPools_ +
					offSolverPools_.size() );

	species_.resize( numVarPools_ + numBufPools_ + numFuncPools_, 0 );
	rates_.resize( numReac_ );
	// v_.resize( numReac_, 0.0 ); // v is now allocated dynamically
	funcs_.resize( numFuncPools_ );
	N_.setSize( idMap_.size(), numReac_ );
}

/// Calculate sizes of all arrays, and allocate them.
void Stoich::allocateModel( const vector< Id >& elist )
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
	for ( vector< Id >::const_iterator 
				i = offSolverReacs_.begin(); 
				i != offSolverReacs_.end(); ++i )
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

	unsigned int numOffSolverPools = idMap_.size();
	for ( vector< Id >::const_iterator i = offSolverPools_.begin(); 
					i != offSolverPools_.end(); ++i ) {
		objMap_[ i->value() - objMapStart_ ] = numOffSolverPools++;
		idMap_.push_back( *i );
	}


	assert( idMap_.size() == numOffSolverPools );
	numFuncPools_ -= numVarPools_ + numBufPools_;
	numVarPoolsBytes_ = numVarPools_ * sizeof( double );

	resizeArrays();
}

void Stoich::installAndUnschedFunc( Id func, Id Pool )
{
	// Unsched Func
	vector< ObjId > unsched( 1, func );
	Shell::dropClockMsgs(  unsched, "process" );

	// Install the FuncTerm
	static const Finfo* funcSrcFinfo = 
			FuncBase::initCinfo()->findFinfo( "input" );
	FuncBase* fb = reinterpret_cast< FuncBase* >( func.eref().data() );
	FuncTerm* ft = fb->func();
	vector< Id > srcPools;
	unsigned int numSrc = func.element()->getNeighbours( 
					srcPools, funcSrcFinfo );
	assert( numSrc > 0 );
	vector< unsigned int > poolIndex( numSrc, 0 );
	for ( unsigned int i = 0; i < numSrc; ++i )
		poolIndex[i] = convertIdToPoolIndex( srcPools[i] );
	ft->setReactants( poolIndex );
	unsigned int funcIndex = convertIdToFuncIndex( func );
	funcs_[ funcIndex ] = ft;
	// Somewhere I have to tie the output of the FuncTerm to the funcPool.
}

// e is the stoich Eref, elist is list of all Ids to zombify.
void Stoich::zombifyModel( const Eref& e, const vector< Id >& elist )
{
	static const Cinfo* poolCinfo = Cinfo::find( "Pool" );
	static const Cinfo* bufPoolCinfo = Cinfo::find( "BufPool" );
	static const Cinfo* funcPoolCinfo = Cinfo::find( "FuncPool" );
	static const Cinfo* reacCinfo = Cinfo::find( "Reac" );
	static const Cinfo* enzCinfo = Cinfo::find( "Enz" );
	static const Cinfo* mmEnzCinfo = Cinfo::find( "MMenz" );
	static const Cinfo* zombiePoolCinfo = Cinfo::find( "ZombiePool" );
	static const Cinfo* zombieBufPoolCinfo = Cinfo::find( "ZombieBufPool");
	static const Cinfo* zombieFuncPoolCinfo = Cinfo::find( "ZombieFuncPool");
	static const Cinfo* zombieReacCinfo = Cinfo::find( "ZombieReac");
	static const Cinfo* zombieMMenzCinfo = Cinfo::find( "ZombieMMenz");
	static const Cinfo* zombieEnzCinfo = Cinfo::find( "ZombieEnz");
	vector< Id > meshEntries;
	vector< Id > temp = elist;

	temp.insert( temp.end(), offSolverReacs_.begin(), offSolverReacs_.end() );

	for ( vector< Id >::const_iterator i = temp.begin(); i != temp.end(); ++i ){
		vector< ObjId > unsched( 1, *i );
		Shell::dropClockMsgs( unsched, "process" );
		Element* ei = i->element();
		if ( ei->cinfo() == poolCinfo ) {
			PoolBase::zombify( i->element(), zombiePoolCinfo, 
							poolInterface_ );
		}
		else if ( ei->cinfo() == bufPoolCinfo ) {
			PoolBase::zombify( i->element(), zombieBufPoolCinfo, 
							poolInterface_ );
		}
		else if ( ei->cinfo() == funcPoolCinfo ) {
			PoolBase::zombify( i->element(), zombieFuncPoolCinfo, 
							poolInterface_ );
			// Has also got to zombify the Func.
			Id funcId = Neutral::child( i->eref(), "func" );
			assert( funcId != Id() );
			assert( funcId.element()->cinfo()->isA( "FuncBase" ) );
			installAndUnschedFunc( funcId, (*i) );
		}
		else if ( ei->cinfo() == reacCinfo ) {
			ReacBase::zombify( ei, zombieReacCinfo, e.id() );
		}
		else if ( ei->cinfo() == mmEnzCinfo ) {
			EnzBase::zombify( ei, zombieMMenzCinfo, e.id() );
		}
		else if ( ei->cinfo() == enzCinfo ) {
			CplxEnzBase::zombify( ei, zombieEnzCinfo, e.id() );
		}
	}
}

void Stoich::unZombifyPools()
{
	static const Cinfo* poolCinfo = Cinfo::find( "Pool" );
	static const Cinfo* bufPoolCinfo = Cinfo::find( "BufPool" );
	static const Cinfo* zombiePoolCinfo = Cinfo::find( "ZombiePool" );
	static const Cinfo* zombieBufPoolCinfo = Cinfo::find( "ZombieBufPool");
	unsigned int i = 0;
	for ( ; i < numVarPools_; ++i ) {
		Element* e = idMap_[i].element();
		if ( e != 0 &&  e->cinfo() == zombiePoolCinfo )
			PoolBase::zombify( e, poolCinfo, Id() );
	}
	
	for ( ; i < numVarPools_ + numBufPools_; ++i ) {
		Element* e = idMap_[i].element();
		if ( e != 0 &&  e->cinfo() == zombieBufPoolCinfo )
			PoolBase::zombify( e, bufPoolCinfo, Id() );
	}
}

void Stoich::unZombifyFuncs()
{
	static const Cinfo* funcPoolCinfo = Cinfo::find( "FuncPool" );
	static const Cinfo* zombieFuncPoolCinfo = Cinfo::find( "ZombieFuncPool");
	Shell* s = reinterpret_cast< Shell* >( Id().eref().data() );
	unsigned int start = numVarPools_ + numBufPools_;
	vector< ObjId > list;
	for ( unsigned int k = 0; k < numFuncPools_; ++k ) {
		unsigned int i = k + start;
		Element* e = idMap_[i].element();
		if ( e != 0 &&  e->cinfo() == zombieFuncPoolCinfo ) {
			PoolBase::zombify( e, funcPoolCinfo, Id() );
			// Has also got to unzombify the Func.
			Id funcId = Neutral::child( idMap_[i].eref(), "func" );
			if ( funcId != Id() ) {
				assert ( funcId.element()->cinfo()->isA( "FuncBase" ) );
				// s->doUseClock( funcId.path(), "process", 5 );
				list.push_back( funcId );
			}
		}
	}
	s->addClockMsgs( list, "process", 5, 0 );
}

void Stoich::unZombifyModel()
{
	static const Cinfo* reacCinfo = Cinfo::find( "Reac" );
	static const Cinfo* enzCinfo = Cinfo::find( "Enz" );
	static const Cinfo* mmEnzCinfo = Cinfo::find( "MMenz" );
	static const Cinfo* zombieReacCinfo = Cinfo::find( "ZombieReac");
	static const Cinfo* zombieMMenzCinfo = Cinfo::find( "ZombieMMenz");
	static const Cinfo* zombieEnzCinfo = Cinfo::find( "ZombieEnz");
	assert (idMap_.size() == numVarPools_ + numBufPools_ + numFuncPools_ +
					offSolverPools_.size() );

	unZombifyPools();
	unZombifyFuncs();

	Shell* s = reinterpret_cast< Shell* >( Id().eref().data() );

	for ( vector< Id >::iterator i = reacMap_.begin(); 
						i != reacMap_.end(); ++i ) {
		Element* e = i->element();
		if ( e != 0 &&  e->cinfo() == zombieReacCinfo )
			ReacBase::zombify( e, reacCinfo, Id() );
	}
	
	for ( vector< Id >::iterator i = mmEnzMap_.begin(); 
						i != mmEnzMap_.end(); ++i ) {
		Element* e = i->element();
		if ( e != 0 &&  e->cinfo() == zombieMMenzCinfo )
			EnzBase::zombify( e, mmEnzCinfo, Id() );
	}
	
	for ( vector< Id >::iterator i = enzMap_.begin(); 
						i != enzMap_.end(); ++i ) {
		Element* e = i->element();
		if ( e != 0 &&  e->cinfo() == zombieEnzCinfo )
			CplxEnzBase::zombify( e, enzCinfo, Id() );
	}

	vector< ObjId > temp( idMap_.begin(), 
			idMap_.begin() + numVarPools_ + numBufPools_ + numFuncPools_ );
	s->addClockMsgs( temp, "proc", 4, 0 );
}

unsigned int Stoich::convertIdToPoolIndex( Id id ) const
{
	unsigned int i = id.value() - objMapStart_;
	assert( i < objMap_.size() );
	i = objMap_[i];
	assert( i < idMap_.size() );
	return i;
}

unsigned int Stoich::convertIdToReacIndex( Id id ) const
{
	unsigned int i = id.value() - objMapStart_;
	assert( i < objMap_.size() );
	i = objMap_[i];
	if ( i < rates_.size() )
			return i;
	return ~0U;
	// assert( i < rates_.size() );
	// return i;
}

unsigned int Stoich::convertIdToFuncIndex( Id id ) const
{
	unsigned int i = id.value() - objMapStart_;
	assert( i < objMap_.size() );
	i = objMap_[i];
	assert( i < funcs_.size() );
	return i;
}

ZeroOrder* makeHalfReaction(
	double rate, const Stoich* sc, const vector< Id >& reactants )
{
	ZeroOrder* rateTerm = 0;
	if ( reactants.size() == 1 ) {
		rateTerm =  new FirstOrder( 
			rate, sc->convertIdToPoolIndex( reactants[0] ) );
	} else if ( reactants.size() == 2 ) {
		rateTerm = new SecondOrder( rate,
			sc->convertIdToPoolIndex( reactants[0] ),
			sc->convertIdToPoolIndex( reactants[1] ) );
	} else if ( reactants.size() > 2 ) {
		vector< unsigned int > temp;
		for ( unsigned int i = 0; i < reactants.size(); ++i )
			temp.push_back( sc->convertIdToPoolIndex( reactants[i] ) );
		rateTerm = new NOrder( rate, temp );
	} else {
		cout << "Error: GslStoichZombies::makeHalfReaction: no reactants\n";
	}
	return rateTerm;
}

/**
 * This takes the specified forward and reverse half-reacs belonging
 * to the specified Reac, and builds them into the Stoich.
 */
void Stoich::installReaction( Id reacId,
		const vector< Id >& subs, 
		const vector< Id >& prds )
{
	ZeroOrder* forward = makeHalfReaction( 0, this, subs );
	ZeroOrder* reverse = makeHalfReaction( 0, this, prds );
	installReaction( reacId, forward, reverse );
}

void Stoich::installReaction( Id reacId, 
				ZeroOrder* forward, ZeroOrder* reverse )
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

/**
 * This takes the baseclass for an MMEnzyme and builds the
 * MMenz into the Stoich.
 */
void Stoich::installMMenz( Id enzId, Id enzMolId,
	const vector< Id >& subs, const vector< Id >& prds )
{
	MMEnzymeBase* meb;
	unsigned int enzIndex = convertIdToPoolIndex( enzMolId );
	unsigned int enzSiteIndex = convertIdToReacIndex( enzId );
	if ( subs.size() == 1 ) {
		unsigned int subIndex = convertIdToPoolIndex( subs[0] );
		meb = new MMEnzyme1( 1, 1, enzIndex, subIndex );
	} else if ( subs.size() > 1 ) {
		vector< unsigned int > v;
		for ( unsigned int i = 0; i < subs.size(); ++i )
			v.push_back( convertIdToPoolIndex( subs[i] ) );
		ZeroOrder* rateTerm = new NOrder( 1.0, v );
		meb = new MMEnzyme( 1, 1, enzIndex, rateTerm );
	} else {
		cout << "Error: GslStoich::installEnzyme: No substrates for "  <<
			enzId.path() << endl;
		return;
	}
	installMMenz( meb, enzSiteIndex, subs, prds );
}

/// This is the internal variant to install the MMenz.
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


void Stoich::installEnzyme( Id enzId, Id enzMolId, Id cplxId,
	const vector< Id >& subs, const vector< Id >& prds )
{
	vector< Id > temp( subs );
	temp.insert( temp.begin(), enzMolId );
	ZeroOrder* r1 = makeHalfReaction( 0, this, temp );
	temp.clear();
	temp.resize( 1, cplxId );
	ZeroOrder* r2 = makeHalfReaction( 0, this, temp );
	ZeroOrder* r3 = makeHalfReaction( 0, this, temp );

	installEnzyme( r1, r2, r3, enzId, enzMolId, prds );
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
	static const Cinfo* zombieReacCinfo = Cinfo::find( "ZombieReac");
	static const SrcFinfo* subOut = dynamic_cast< const SrcFinfo* > (
		zombieReacCinfo->findFinfo( "subOut" ) );
	assert( subOut );

	double volScale = convertConcToNumRateUsingMesh( e, subOut, false );
	unsigned int i = convertIdToReacIndex( e.id() );
	if ( i != ~0U )
		rates_[ i ]->setR1( v / volScale );
}

/**
 * For now assume a single rate term.
 */
void Stoich::setReacKb( const Eref& e, double v ) const
{
	static const Cinfo* zombieReacCinfo = Cinfo::find( "ZombieReac");
	static const SrcFinfo* prdOut = static_cast< const SrcFinfo* > (
		zombieReacCinfo->findFinfo( "prdOut" ) );
	static const SrcFinfo* subOut = dynamic_cast< const SrcFinfo* > (
		zombieReacCinfo->findFinfo( "subOut" ) );

	assert( prdOut );
	assert( subOut );
	double volScale = convertConcToNumRateUsingMesh( e, prdOut, false );
	// cout << "Id=" << e.id() << ", Kb = " << v << ", volScale = " << volScale << endl;
	/*
	 * This rescaling is now done in the convertConc func itself.
	// Assume first substrate is the reference volume. Scale down by this.
	vector< double > vols;
	getReactantVols( e, subOut, vols );
	assert( vols.size() > 0 );
	assert( vols[0] > 0.0 );
	volScale /= vols[0] * NA;
	*/
	unsigned int i = convertIdToReacIndex( e.id() );
	if ( i == ~0U )
		return;

	if ( useOneWay_ )
		 rates_[ i + 1 ]->setR1( v / volScale);
	else
		 rates_[ i ]->setR2( v / volScale );
}

void Stoich::setMMenzKm( const Eref& e, double v ) const
{
	static const Cinfo* zombieMMenzCinfo = Cinfo::find( "ZombieMMenz");
	static const SrcFinfo* subOut = dynamic_cast< const SrcFinfo* > (
		zombieMMenzCinfo->findFinfo( "subOut" ) );
	// Identify MMenz rate term
	RateTerm* rt = rates_[ convertIdToReacIndex( e.id() ) ];
	MMEnzymeBase* enz = dynamic_cast< MMEnzymeBase* >( rt );
	assert( enz );
	// Identify MMenz Enzyme substrate. I would have preferred the parent,
	// but that gets messy.
	// unsigned int enzMolIndex = enz->getEnzIndex();

	// This function can be replicated to handle multiple different voxels.
	vector< double > vols;
	getReactantVols( e, subOut, vols );
	if ( vols.size() == 0 ) {
		cerr << "Error: Stoich::setMMenzKm: no substrates for enzyme " <<
			e << endl;
		return;
	}
	// Do scaling and assignment.
	enz->setR1( v * vols[0] * NA );
}

double Stoich::getMMenzNumKm( const Eref& e ) const
{
	return getR1( e );
}

void Stoich::setMMenzKcat( const Eref& e, double v ) const
{
	RateTerm* rt = rates_[ convertIdToReacIndex( e.id() ) ];
	MMEnzymeBase* enz = dynamic_cast< MMEnzymeBase* >( rt );
	assert( enz );

	enz->setR2( v );
}

double Stoich::getMMenzKcat( const Eref& e ) const
{
	return getR2( e );
}

/// Later handle all the volumes when this conversion is done.
void Stoich::setEnzK1( const Eref& e, double v ) const
{
	static const Cinfo* zombieEnzCinfo = Cinfo::find( "ZombieEnz");
	static const SrcFinfo* subOut = dynamic_cast< const SrcFinfo* > (
		zombieEnzCinfo->findFinfo( "subOut" ) );
	assert( subOut );

	double volScale = convertConcToNumRateUsingMesh( e, subOut, true );

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

double Stoich::getEnzNumK1( const Eref& e ) const
{
	return getR1( e );
}

double Stoich::getEnzK2( const Eref& e ) const
{
	if ( useOneWay_ )
		return getR1offset1( e );
	else
		return getR2( e );
}

double Stoich::getEnzK3( const Eref& e ) const
{
	if ( useOneWay_ )
		return getR1offset2( e );
	else
		return getR1offset1( e );
}

/**
 * Looks up the matching rate for R1. Later we may have additional 
 * scaling terms for the specified voxel.
 */
double Stoich::getR1( const Eref& e ) const
{
	return rates_[ convertIdToReacIndex( e.id() ) ]->getR1();
}
double Stoich::getR1offset1( const Eref& e ) const
{
	return rates_[ convertIdToReacIndex( e.id() ) + 1 ]->getR1();
}
double Stoich::getR1offset2( const Eref& e ) const
{
	return rates_[ convertIdToReacIndex( e.id() ) + 2 ]->getR1();
}

/**
 * Looks up the matching rate for R2. Later we may have additional 
 * scaling terms for the specified voxel.
 */
double Stoich::getR2( const Eref& e ) const
{
	return rates_[ convertIdToReacIndex( e.id() ) ]->getR2();
}

SpeciesId Stoich::getSpecies( unsigned int poolIndex ) const
{
	return species_[ poolIndex ];
}

void Stoich::setSpecies( unsigned int poolIndex, SpeciesId s )
{
	species_[ poolIndex ] = s;
}

// for debugging.
void Stoich::print() const
{
	N_.print();
}

// for debugging
void Stoich::printRates() const
{
	for ( vector< Id >::const_iterator 
		i = reacMap_.begin(); i != reacMap_.end(); ++i ) {
		double Kf = Field< double >::get( *i, "Kf");
		double Kb = Field< double >::get( *i, "Kb");
		double kf = Field< double >::get( *i, "kf");
		double kb = Field< double >::get( *i, "kb");
			cout << "Id=" << *i << ", (Kf,Kb) = (" << Kf << ", " << Kb <<
				  "), (kf, kb) = (" << kf << ", " << kb << ")\n";
	}
}

/////////////////////////////////////////////////////////////////////
const vector< Id >& Stoich::getOffSolverPools() const
{
	return offSolverPools_;
}

vector< Id > Stoich::getOffSolverCompts() const
{
	vector< Id > ret;
	for ( map< Id, vector< Id > >::const_iterator 
		i = offSolverPoolMap_.begin(); i != offSolverPoolMap_.end(); ++i )
			ret.push_back( i->first );

	return ret;
}

const vector< Id >& Stoich::offSolverPoolMap( Id compt ) const
{
	static vector< Id > blank( 0 );
	map< Id, vector < Id > >::const_iterator i = 
			offSolverPoolMap_.find( compt );
	if ( i != offSolverPoolMap_.end() )
		return i->second;
	return blank;
}

/////////////////////////////////////////////////////////////////////
// Numeric funcs. These are in Stoich because the rate terms are here.
/////////////////////////////////////////////////////////////////////

/**
 * updateRates computes the velocity *v* of each reaction. Then it
 * uses this to compute the rate of change, *yprime*, for each pool
 */

void Stoich::updateRates( const double* s, double* yprime )
{
	vector< RateTerm* >::const_iterator i;
	vector< double > v( numReac_, 0.0 );
	vector< double >::iterator j = v.begin();
	assert( numReac_ == rates_.size() );

	for ( i = rates_.begin(); i != rates_.end(); i++) {
		*j++ = (**i)( s );
		assert( !isnan( *( j-1 ) ) );
	}

	for (unsigned int i = 0; i < numVarPools_ + offSolverPools_.size(); ++i)
		*yprime++ = N_.computeRowRate( i , v );
}

// s is the array of pools, S_[meshIndex][0]
void Stoich::updateFuncs( double* s, double t )
{
	double* j = s + numVarPools_ + numBufPools_;

	for ( vector< FuncTerm* >::iterator i = funcs_.begin();
					i != funcs_.end(); ++i ) {
		*j++ = (**i)( s, t );
		assert( !isnan( *(j-1) ) );
	}
}

/**
 * updateJunctionRates:
 * Updates the rates for cross-compartment reactions. These are located
 * at the end of the rates_ vector, and are directly indexed by the
 * reacTerms.
 */
void Stoich::updateJunctionRates( const double* s,
	const vector< unsigned int >& reacTerms, double* yprime )
{
	for ( vector< unsigned int >::const_iterator i = reacTerms.begin(); 
					i != reacTerms.end(); ++i )
	{
			assert( *i < rates_.size() );
			*yprime++ += (*rates_[*i])( s );
	}
}

void Stoich::updateRatesAfterRemesh()
{
	for ( vector< Id >::iterator 
					i = reacMap_.begin(); i != reacMap_.end(); ++i ) {
		double Kf = Field< double >::get( *i, "Kf");
		double Kb = Field< double >::get( *i, "Kb");
		setReacKf( i->eref(), Kf );
		setReacKb( i->eref(), Kb );
	}
	for ( vector< Id >::iterator 
			i = offSolverReacs_.begin(); i != offSolverReacs_.end(); ++i ){
		if ( i->element()->cinfo()->isA( "ReacBase" ) ) {
			double Kf = Field< double >::get( *i, "Kf");
			double Kb = Field< double >::get( *i, "Kb");
			setReacKf( i->eref(), Kf );
			setReacKb( i->eref(), Kb );
		} else if ( i->element()->cinfo()->isA( "CplxEnzBase" ) ) {
			double concK1 = Field< double >::get( *i, "concK1");
			double k3 = Field< double >::get( *i, "k3");
			double k2 = Field< double >::get( *i, "k2");
			setEnzK3( i->eref(), k3 );
			setEnzK2( i->eref(), k2 );
			setEnzK1( i->eref(), concK1 );
		} else if ( i->element()->cinfo()->isA( "MMEnzBase" ) ) {
			double Km = Field< double >::get( *i, "Km");
			double kcat = Field< double >::get( *i, "kcat");
			setMMenzKm( i->eref(), Km );
			setMMenzKcat( i->eref(), kcat );
		}
	}
}


