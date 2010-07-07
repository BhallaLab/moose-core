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
#include "RateTerm.h"
#include "SparseMatrix.h"
#include "KinSparseMatrix.h"
#include "Stoich.h"
#include "Mol.h"
#include "BufMol.h"
#include "Reac.h"
#include "Enz.h"
#include "MMenz.h"
#include "ZombieMol.h"
#include "ZombieBufMol.h"
#include "ZombieReac.h"
#include "ZombieEnz.h"
#include "ZombieMMenz.h"

#ifdef USE_GSL
#include <gsl/gsl_errno.h>
#endif

#define EPSILON 1e-15

static SrcFinfo1< Stoich* > plugin( 
		"plugin", 
		"Sends out Stoich pointer so that plugins can directly access fields and functions"
	);

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

		static ReadOnlyValueFinfo< Stoich, double > nVarMols(
			"nVarMols",
			"Number of variable molecules in the reac system",
			&Stoich::getNumVarMols
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
		static DestFinfo process( "process",
			"Handles process call",
			new ProcOpFunc< Stoich >( &Stoich::process ) );
		static DestFinfo reinit( "reinit",
			"Handles reinint call",
			new ProcOpFunc< Stoich >( &Stoich::reinit ) );

		//////////////////////////////////////////////////////////////
		// SharedMsg Definitions
		//////////////////////////////////////////////////////////////
		static Finfo* procShared[] = {
			&process, &reinit
		};
		static SharedFinfo proc( "proc",
			"Shared message for process and reinit",
			procShared, sizeof( procShared ) / sizeof( const Finfo* )
		);

	static Finfo* stoichFinfos[] = {
		&useOneWay,		// Value
		&nVarMols,		// Value
		&path,			// Value
		&plugin,		// SrcFinfo
		&proc,			// SharedFinfo
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
		objMapStart_( 0 ),
		numVarMols_( 0 ),
		numVarMolsBytes_( 0 ),
		numReac_( 0 )
{;}

Stoich::~Stoich()
{
	for ( vector< RateTerm* >::iterator i = rates_.begin();
		i != rates_.end(); ++i )
		delete *i;
}

//////////////////////////////////////////////////////////////
// MsgDest Definitions
//////////////////////////////////////////////////////////////

void Stoich::process( const Eref& e, ProcPtr p )
{
	;
}

void Stoich::reinit( const Eref& e, ProcPtr p )
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

	/*
	cout << "Zombified " << numVarMols_ << " Molecules, " <<
		numReac_ << " reactions\n";
	N_.print();
	*/
}

string Stoich::getPath( const Eref& e, const Qinfo* q ) const
{
	return path_;
}

double Stoich::getNumVarMols() const
{
	return numVarMols_;
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
	static const Cinfo* molCinfo = Mol::initCinfo();
	static const Cinfo* bufMolCinfo = BufMol::initCinfo();
	static const Cinfo* reacCinfo = Reac::initCinfo();
	static const Cinfo* enzCinfo = Enz::initCinfo();
	static const Cinfo* mmEnzCinfo = MMenz::initCinfo();
	numVarMols_ = 0;
	numReac_ = 0;
	vector< Id > bufMols;
	for ( vector< Id >::const_iterator i = elist.begin(); i != elist.end(); ++i ){
		Element* ei = (*i)();
		if ( ei->cinfo() == molCinfo ) {
			objMap_[ i->value() - objMapStart_ ] = numVarMols_;
			++numVarMols_;
		}
		if ( ei->cinfo() == bufMolCinfo ) {
			bufMols.push_back( *i );
		}
		if ( ei->cinfo() == reacCinfo || ei->cinfo() == mmEnzCinfo ) {
			objMap_[ i->value() - objMapStart_ ] = numReac_;
			++numReac_;
		}
		if ( ei->cinfo() == enzCinfo ) {
			objMap_[ i->value() - objMapStart_ ] = numReac_;
			numReac_ += 2;
		}
	}

	numBufMols_ = 0;
	for ( vector< Id >::const_iterator i = bufMols.begin(); i != bufMols.end(); ++i ){
		objMap_[ i->value() - objMapStart_ ] = numVarMols_ + numBufMols_;
		++numBufMols_;
	}

	numVarMolsBytes_ = numVarMols_ * sizeof( double );
	S_.resize( numVarMols_ + numBufMols_, 0.0 );
	Sinit_.resize( numVarMols_ + numBufMols_, 0.0 );
	rates_.resize( numReac_ );
	v_.resize( numReac_, 0.0 );
	N_.setSize( numVarMols_ + numBufMols_, numReac_ );
}

void Stoich::zombifyModel( const Eref& e, const vector< Id >& elist )
{
	static const Cinfo* molCinfo = Mol::initCinfo();
	static const Cinfo* bufMolCinfo = BufMol::initCinfo();
	static const Cinfo* reacCinfo = Reac::initCinfo();
	static const Cinfo* enzCinfo = Enz::initCinfo();
	static const Cinfo* mmEnzCinfo = MMenz::initCinfo();

	for ( vector< Id >::const_iterator i = elist.begin(); i != elist.end(); ++i ){
		Element* ei = (*i)();
		if ( ei->cinfo() == molCinfo ) {
			ZombieMol::zombify( e.element(), (*i)() );
		}
		else if ( ei->cinfo() == bufMolCinfo ) {
			ZombieBufMol::zombify( e.element(), (*i)() );
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
	}
}

unsigned int Stoich::convertIdToMolIndex( Id id ) const
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

	for ( i = rates_.begin(); i != rates_.end(); i++)
	{
		*j++ = (**i)();
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
	assert( yprime->size() >= numVarMols_ );
	for (unsigned int i = 0; i < numVarMols_; i++) {
		*j++ = dt * N_.computeRowRate( i , v_ );
	}
}

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
	//nCall_++;
//	if ( lasty_ != y ) { // should count to see how often this copy happens
		// Copy the y array into the y_ vector.
		memcpy( &S_[0], y, numVarMolsBytes_ );
		// lasty_ = y;
	//	nCopy_++;
//	}

//	updateDynamicBuffers();

	updateV();

	// Much scope for optimization here.
	for (unsigned int i = 0; i < numVarMols_; i++) {
		*yprime++ = N_.computeRowRate( i , v_ );
	}
	// cout << t << ": " << y[0] << ", " << y[1] << endl;
	return GSL_SUCCESS;
}

#endif // USE_GSL
