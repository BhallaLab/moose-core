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

		static LookupValueFinfo< Stoich, short, double > compartmentVolume(
			"compartmentVolume",
			"Size of specified compartment",
			&Stoich::setCompartmentVolume,
			&Stoich::getCompartmentVolume
		);

		static ElementValueFinfo< Stoich, string > path(
			"path",
			"Path of reaction system to take over",
			&Stoich::setPath,
			&Stoich::getPath
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

		//////////////////////////////////////////////////////////////
		// FieldElementFinfo defintion for Ports.
		//////////////////////////////////////////////////////////////
		static FieldElementFinfo< Stoich, Port > portFinfo( "port",
			"Sets up field Elements for ports",
			Port::initCinfo(),
			&Stoich::getPort,
			&Stoich::setNumPorts,
			&Stoich::getNumPorts
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
		&compartmentVolume,	//Value
		&path,			// Value
		plugin(),		// SrcFinfo
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
void Stoich::innerReinit()
{
	y_.assign( Sinit_.begin(), Sinit_.begin() + numVarPools_ );
	S_ = Sinit_;

	updateFuncs( 0 );
	updateV();
}

/**
 * Handles incoming messages representing influx of molecules
 */
void Stoich::influx( DataId port, vector< double > pool )
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
void Stoich::handleAvailableMolsAtPort( DataId port, vector< SpeciesId > mols )
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

void Stoich::handleMatchedMolsAtPort( DataId port, vector< SpeciesId > mols )
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
	y_.assign( Sinit_.begin(), Sinit_.begin() + numVarPools_ );
	S_ = Sinit_;

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

unsigned int Stoich::numCompartments() const
{
	return compartment_.size();
}

void Stoich::setCompartmentVolume( short comptIndex, double v )
{
	if ( v <= 0 ) {
		cout << "Error: Stoich::setCompartmentVolume: volume v must be > 0\n";
		return;
	}
	unsigned int ci = comptIndex;
	if ( ci >= compartmentSize_.size() ) {
		cout << "Error: Stoich::setCompartmentVolume: Index " <<
			comptIndex << " out of range, only " << 
				compartmentSize_.size() <<
			" compartments present\n";
		return;
	}

	double origVol = compartmentSize_[ comptIndex ];
	double ratio = v/origVol;

	assert( compartment_.size() == S_.size() );
	assert( compartment_.size() == Sinit_.size() );
	for ( unsigned int i = 0; i < compartment_.size(); ++i ) {
		S_[i] *= ratio;
		Sinit_[i] *= ratio;
	}
	for ( vector< double >::iterator i = y_.begin(); i != y_.end(); ++i )
		*i *= ratio;

	for ( vector< RateTerm* >::iterator i = rates_.begin(); i != rates_.end(); ++i ) {
		(*i)->rescaleVolume(  comptIndex , compartment_, ratio );
	}
}

double Stoich::getCompartmentVolume( short i ) const
{
	unsigned int temp = i;
	if ( temp < compartmentSize_.size() )
		return compartmentSize_[i];
	return 0.0;
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

	/*
	for ( unsigned int i = 0; i < elist.size(); ++i ) {
		unsigned int index = elist[i].value() - objMapStart_;
		objMap_[ index ] = i;
	}
	*/
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
	numVarPools_ = 0;
	numReac_ = 0;
	vector< Id > bufPools;
	vector< Id > funcPools;
	unsigned int numFunc = 0;
	for ( vector< Id >::const_iterator i = elist.begin(); i != elist.end(); ++i ){
		Element* ei = (*i)();
		if ( ei->cinfo() == poolCinfo ) {
			objMap_[ i->value() - objMapStart_ ] = numVarPools_;
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
	// numVarPools_ += numEfflux_;

	numBufPools_ = 0;
	for ( vector< Id >::const_iterator i = bufPools.begin(); i != bufPools.end(); ++i ){
		objMap_[ i->value() - objMapStart_ ] = numVarPools_ + numBufPools_;
		++numBufPools_;
	}

	numFuncPools_ = numVarPools_ + numBufPools_;
	for ( vector< Id >::const_iterator i = funcPools.begin(); 
		i != funcPools.end(); ++i ) {
		objMap_[ i->value() - objMapStart_ ] = numFuncPools_++;
	}
	numFuncPools_ -= numVarPools_ + numBufPools_;
	assert( numFunc == numFuncPools_ );

	numVarPoolsBytes_ = numVarPools_ * sizeof( double );
	S_.resize( numVarPools_ + numBufPools_ + numFuncPools_, 0.0 );
	Sinit_.resize( numVarPools_ + numBufPools_ + numFuncPools_, 0.0 );
	compartment_.resize( numVarPools_ + numBufPools_ + numFuncPools_, 0 );
	species_.resize( numVarPools_ + numBufPools_ + numFuncPools_, 0 );
	y_.resize( numVarPools_ );
	rates_.resize( numReac_ );
	v_.resize( numReac_, 0.0 );
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
		else if ( ei->cinfo()->isA( "MeshEntry" ) ) {
			zombifyChemMesh( *i ); // It retains its identity.
			// ZombieChemMesh::zombify( e.element(), (*i)() );
		}
	}
}

void Stoich::zombifyChemMesh( Id meshEntry )
{
	static const Cinfo* meshEntryCinfo = MeshEntry::initCinfo();
	static const Finfo* finfo = meshEntryCinfo->findFinfo( "get_size" );

	MeshEntry* c = reinterpret_cast< MeshEntry* >( meshEntry.eref().data());

	Element* e = meshEntry();
	vector< Id > pools;
	const DestFinfo* df = dynamic_cast< const DestFinfo* >( finfo );
	assert( df );
	unsigned int numTgts = e->getInputs( pools, df );
	assert( numTgts > 0 );

	for ( vector< Id >::iterator i = pools.begin(); i != pools.end(); ++i ){
		unsigned int m = convertIdToPoolIndex( *i );
		compartment_[ m ] = compartmentSize_.size();
	}

	objMap_[ meshEntry.value() - objMapStart_ ] = compartmentSize_.size();
	compartmentSize_.push_back( c->getSize( meshEntry.eref(), 0 ) );
}

unsigned int Stoich::convertIdToPoolIndex( Id id ) const
{
	unsigned int i = id.value() - objMapStart_;
	assert( i < objMap_.size() );
	i = objMap_[i];
	assert( i < S_.size() );
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

unsigned int Stoich::convertIdToComptIndex( Id id ) const
{
	unsigned int i = id.value() - objMapStart_;
	assert( i < objMap_.size() );
	i = objMap_[i];
	assert( i < compartmentSize_.size() );
	return i;
}

//////////////////////////////////////////////////////////////
// Model running functions
//////////////////////////////////////////////////////////////

// Update the v_ vector for individual reac velocities.
void Stoich::updateV( )
{
	// Some algorithm to assign the values from the computed rates
	// to the corresponding v_ vector entry
	// for_each( rates_.begin(), rates_.end(), assign);

	vector< RateTerm* >::const_iterator i;
	vector< double >::iterator j = v_.begin();
	const double* S = &S_[0];

	for ( i = rates_.begin(); i != rates_.end(); i++)
	{
		*j++ = (**i)( S );
		assert( !isnan( *( j-1 ) ) );
	}

	// I should use foreach here.
	/*
	vector< SumTotal >::const_iterator k;
	for ( k = sumTotals_.begin(); k != sumTotals_.end(); k++ )
		k->sum();
	*/
}

void Stoich::updateRates( vector< double>* yprime, double dt  )
{
	updateV();

	// Much scope for optimization here.
	vector< double >::iterator j = yprime->begin();
	assert( yprime->size() >= numVarPools_ );
	for (unsigned int i = 0; i < numVarPools_; i++) {
		*j++ = dt * N_.computeRowRate( i , v_ );
	}
}

// Update the function-computed molecule terms. These are not integrated,
// but their values may be used by molecules that are.
// The molecule vector S_ has a section for FuncTerms. In this section
// there is a one-to-one match between entries in S_ and FuncTerm entries.
void Stoich::updateFuncs( double t )
{
	vector< FuncTerm* >::const_iterator i;
	vector< double >::iterator j = S_.begin() + numVarPools_ + numBufPools_;

	for ( i = funcs_.begin(); i != funcs_.end(); i++)
	{
		*j++ = (**i)( &( S_[0] ), t );
		assert( !isnan( *( j-1 ) ) );
	}
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
const double* Stoich::S() const
{
	return &S_[0];
}

double* Stoich::varS()
{
	return &S_[0];
}

const double* Stoich::Sinit() const
{
	return &Sinit_[0];
}

double* Stoich::getY()
{
	return &y_[0];
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
	return static_cast< Stoich* >( s )->innerGslFunc( t, y, yprime );
}


int Stoich::innerGslFunc( double t, const double* y, double* yprime )
{
	// Copy the y array into the S_ vector.
	// Sometimes GSL passes in its own allocated version of y.
	memcpy( &S_[0], y, numVarPoolsBytes_ );

//	updateDynamicBuffers();
	updateFuncs( t );

	updateV();

	// Much scope for optimization here.
	for (unsigned int i = 0; i < numVarPools_; i++) {
		*yprime++ = N_.computeRowRate( i , v_ );
	}
	// cout << t << ": " << y[0] << ", " << y[1] << endl;
	return GSL_SUCCESS;
}

#endif // USE_GSL
