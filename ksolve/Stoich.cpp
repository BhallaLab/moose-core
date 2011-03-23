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
#include "Mol.h"
#include "BufMol.h"
#include "FuncMol.h"
#include "Reac.h"
#include "Enz.h"
#include "MMenz.h"
#include "SumFunc.h"
#include "Boundary.h"
#include "ChemCompt.h"
#include "ZombieMol.h"
#include "ZombieBufMol.h"
#include "ZombieFuncMol.h"
#include "ZombieReac.h"
#include "ZombieEnz.h"
#include "ZombieMMenz.h"
#include "ZombieSumFunc.h"

#include "ReduceBase.h"
#include "ReduceMax.h"
#include "../shell/Shell.h"

#ifdef USE_GSL
#include <gsl/gsl_errno.h>
#endif

#define EPSILON 1e-15

static SrcFinfo1< Id > plugin( 
		"plugin", 
		"Sends out Stoich Id so that plugins can directly access fields and functions"
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

		static ReadOnlyValueFinfo< Stoich, unsigned int > nVarMols(
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
		totPortSize_( 0 ),
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

	for ( vector< FuncTerm* >::iterator i = funcs_.begin();
		i != funcs_.end(); ++i )
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
	y_.assign( Sinit_.begin(), Sinit_.begin() + numVarMols_ );
	S_ = Sinit_;
}

/**
 * Handles incoming messages representing influx of molecules
 */
void Stoich::influx( DataId port, vector< double > mol )
{
	/*
	assert( mol.size() == inPortEnd_ - inPortStart_ );
	unsigned int j = 0;
	for ( unsigned int i = inPortStart_; i < inPortEnd_; ++i ) {
		S_[i] += mol[j++];
	}
	*/
}

void Stoich::handleAvailableMols( DataId port, vector< Id > mols )
{
	;
}

void Stoich::handleMatchedMols( DataId port, vector< Id > mols )
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
	y_.assign( Sinit_.begin(), Sinit_.begin() + numVarMols_ );

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

unsigned int Stoich::getNumVarMols() const
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
	static const Cinfo* funcMolCinfo = FuncMol::initCinfo();
	static const Cinfo* reacCinfo = Reac::initCinfo();
	static const Cinfo* enzCinfo = Enz::initCinfo();
	static const Cinfo* mmEnzCinfo = MMenz::initCinfo();
	static const Cinfo* sumFuncCinfo = SumFunc::initCinfo();
	numVarMols_ = 0;
	numReac_ = 0;
	vector< Id > bufMols;
	vector< Id > funcMols;
	unsigned int numFunc = 0;
	for ( vector< Id >::const_iterator i = elist.begin(); i != elist.end(); ++i ){
		Element* ei = (*i)();
		if ( ei->cinfo() == molCinfo ) {
			objMap_[ i->value() - objMapStart_ ] = numVarMols_;
			++numVarMols_;
		} else if ( ei->cinfo() == bufMolCinfo ) {
			bufMols.push_back( *i );
		} else if ( ei->cinfo() == funcMolCinfo ) {
			funcMols.push_back( *i );
		} else if ( ei->cinfo() == reacCinfo || ei->cinfo() == mmEnzCinfo ){
			objMap_[ i->value() - objMapStart_ ] = numReac_;
			++numReac_;
		} else if ( ei->cinfo() == enzCinfo ) {
			objMap_[ i->value() - objMapStart_ ] = numReac_;
			numReac_ += 2;
		} else if ( ei->cinfo() == sumFuncCinfo ){
			objMap_[ i->value() - objMapStart_ ] = numFunc;
			++numFunc;
		}
	}
	// numVarMols_ += numEfflux_;

	numBufMols_ = 0;
	for ( vector< Id >::const_iterator i = bufMols.begin(); i != bufMols.end(); ++i ){
		objMap_[ i->value() - objMapStart_ ] = numVarMols_ + numBufMols_;
		++numBufMols_;
	}

	numFuncMols_ = numVarMols_ + numBufMols_;
	for ( vector< Id >::const_iterator i = funcMols.begin(); 
		i != funcMols.end(); ++i ) {
		objMap_[ i->value() - objMapStart_ ] = numFuncMols_++;
	}
	numFuncMols_ -= numVarMols_ + numBufMols_;
	assert( numFunc == numFuncMols_ );

	numVarMolsBytes_ = numVarMols_ * sizeof( double );
	S_.resize( numVarMols_ + numBufMols_ + numFuncMols_, 0.0 );
	Sinit_.resize( numVarMols_ + numBufMols_ + numFuncMols_, 0.0 );
	compartment_.resize( numVarMols_ + numBufMols_ + numFuncMols_, 0.0 );
	y_.resize( numVarMols_ );
	rates_.resize( numReac_ );
	v_.resize( numReac_, 0.0 );
	funcs_.resize( numFuncMols_ );
	N_.setSize( numVarMols_ + numBufMols_ + numFuncMols_, numReac_ );
}

void Stoich::zombifyModel( const Eref& e, const vector< Id >& elist )
{
	static const Cinfo* molCinfo = Mol::initCinfo();
	static const Cinfo* bufMolCinfo = BufMol::initCinfo();
	static const Cinfo* funcMolCinfo = FuncMol::initCinfo();
	static const Cinfo* reacCinfo = Reac::initCinfo();
	static const Cinfo* enzCinfo = Enz::initCinfo();
	static const Cinfo* mmEnzCinfo = MMenz::initCinfo();
	static const Cinfo* chemComptCinfo = ChemCompt::initCinfo();
	// static const Cinfo* sumFuncCinfo = SumFunc::initCinfo();
	// The FuncMol handles zombification of stuff coming in to it.

	for ( vector< Id >::const_iterator i = elist.begin(); i != elist.end(); ++i ){
		Element* ei = (*i)();
		if ( ei->cinfo() == molCinfo ) {
			ZombieMol::zombify( e.element(), (*i)() );
		}
		else if ( ei->cinfo() == bufMolCinfo ) {
			ZombieBufMol::zombify( e.element(), (*i)() );
		}
		else if ( ei->cinfo() == funcMolCinfo ) {
			ZombieFuncMol::zombify( e.element(), (*i)() );
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
		else if ( ei->cinfo() == chemComptCinfo ) {
			zombifyChemCompt( *i ); // It retains its identity.
			// ZombieChemCompt::zombify( e.element(), (*i)() );
		}
	}
}

void Stoich::zombifyChemCompt( Id compt )
{
	static const Cinfo* chemComptCinfo = ChemCompt::initCinfo();
	static const Finfo* finfo = chemComptCinfo->findFinfo( "compartment" );
	ChemCompt* c = reinterpret_cast< ChemCompt* >( compt.eref().data() );

	Element* e = compt();
	vector< Id > mols;
	const SrcFinfo* sf = dynamic_cast< const SrcFinfo* >( finfo );
	assert( sf );
	unsigned int numTgts = e->getOutputs( mols, sf );

	for ( vector< Id >::iterator i = mols.begin(); i != mols.end(); ++i ) {
		unsigned int m = convertIdToMolIndex( *i );
		compartment_[ m ] = compartmentSize_.size();
	}

	assert( numTgts > 0 );

	compartmentSize_.push_back( c->getSize() );
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

unsigned int Stoich::convertIdToFuncIndex( Id id ) const
{
	unsigned int i = id.value() - objMapStart_;
	assert( i < objMap_.size() );
	i = objMap_[i];
	assert( i < funcs_.size() );
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
	assert( yprime->size() >= numVarMols_ );
	for (unsigned int i = 0; i < numVarMols_; i++) {
		*j++ = dt * N_.computeRowRate( i , v_ );
	}
}

// Update the function-computed molecule terms. These are not integrated,
// but their values may be used by molecules that are.
// The molecule vector S_ has a section for mathTerms. In this section
// there is a one-to-one match between entries in S_ and MathTerm entries.
void Stoich::updateFuncs( double t )
{
	vector< FuncTerm* >::const_iterator i;
	vector< double >::iterator j = S_.begin() + numVarMols_ + numBufMols_;

	for ( i = funcs_.begin(); i != funcs_.end(); i++)
	{
		*j++ = (**i)( t );
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
	//nCall_++;
//	if ( lasty_ != y ) { // should count to see how often this copy happens
		// Copy the y array into the y_ vector.
		memcpy( &S_[0], y, numVarMolsBytes_ );
		// lasty_ = y;
	//	nCopy_++;
//	}

//	updateDynamicBuffers();
	updateFuncs( t );

	updateV();

	// Much scope for optimization here.
	for (unsigned int i = 0; i < numVarMols_; i++) {
		*yprime++ = N_.computeRowRate( i , v_ );
	}
	// cout << t << ": " << y[0] << ", " << y[1] << endl;
	return GSL_SUCCESS;
}

#endif // USE_GSL
