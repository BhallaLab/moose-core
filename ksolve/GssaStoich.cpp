/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include <map>
#include <algorithm>
#include "moose.h"
#include "RateTerm.h"
#include "KinSparseMatrix.h"
#include "Stoich.h"
#include "GssaStoich.h"
#include "randnum.h"

const Cinfo* initGssaStoichCinfo()
{
	/**
	 * Messages that connect to the KineticIntegrator
	 */
	static Finfo* integrateShared[] =
	{
		new DestFinfo( "reinit", Ftype0::global(),
			&Stoich::reinitFunc ),
		new DestFinfo( "integrate",
			Ftype2< vector< double >* , double >::global(),
			RFCAST( &Stoich::integrateFunc ) ),
		new SrcFinfo( "allocate",
			Ftype1< vector< double >* >::global() ),
	};

	/**
	 * Messages that connect to the GssaIntegrator object
	 */
	static Finfo* gssaShared[] =
	{
		new DestFinfo( "reinit", Ftype0::global(),
			&Stoich::reinitFunc ),
		new SrcFinfo( "assignStoich",
			Ftype1< void* >::global() ),
		new SrcFinfo( "assignY",
			Ftype2< double, unsigned int >::global() ),
	};

	/**
	 * These are the fields of the stoich class
	 */
	static Finfo* stoichFinfos[] =
	{
		///////////////////////////////////////////////////////
		// Field definitions
		///////////////////////////////////////////////////////
		///////////////////////////////////////////////////////
		// MsgSrc definitions
		///////////////////////////////////////////////////////
		///////////////////////////////////////////////////////
		// MsgDest definitions
		///////////////////////////////////////////////////////
		///////////////////////////////////////////////////////
		// Shared definitions
		///////////////////////////////////////////////////////
		new SharedFinfo( "integrate", integrateShared, 
				sizeof( integrateShared )/ sizeof( Finfo* ) ),
		new SharedFinfo( "gssa", gssaShared, 
				sizeof( gssaShared )/ sizeof( Finfo* ) ),
	};

	static Cinfo gssaStoichCinfo(
		"GssaStoich",
		"Upinder S. Bhalla, 2008, NCBS",
		"GssaStoich: Gillespie Stochastic Simulation Algorithm object.\nClosely based on the Stoich object and inherits its \nhandling functions for constructing the matrix. Sets up stoichiometry matrix based calculations from a\nwildcard path for the reaction system.\nKnows how to compute derivatives for most common\nthings, also knows how to handle special cases where the\nobject will have to do its own computation. Generates a\nstoichiometry matrix, which is useful for lots of other\noperations as well.",
		initNeutralCinfo(),
		stoichFinfos,
		sizeof( stoichFinfos )/sizeof(Finfo *),
		ValueFtype1< Stoich >::global()
	);

	return &gssaStoichCinfo;
}

static const Cinfo* gssaStoichCinfo = initGssaStoichCinfo();

///////////////////////////////////////////////////
// Class function definitions
///////////////////////////////////////////////////

GssaStoich::GssaStoich()
	: Stoich()
{
	;
}
		
///////////////////////////////////////////////////
// Field function definitions
///////////////////////////////////////////////////
///////////////////////////////////////////////////
// Dest function definitions
///////////////////////////////////////////////////

// Static func
void GssaStoich::reinitFunc( const Conn* c )
{
	Stoich::reinitFunc( c );
	GssaStoich* s = static_cast< GssaStoich* >( c->data() );
	// Here we round off up or down with prob depending on fractional
	// part of the init value.
	for ( vector< double >::iterator i = s->S_.begin(); 
		i != s->S_.end(); ++i ) {
		double base = floor( *i );
		double frac = *i - base;
		if ( mtrand() < frac )
			*i = base;
		else
			*i = base + 1.0;
	}
}

// static func
void GssaStoich::integrateFunc( const Conn* c, vector< double >* v, double dt )
{
	// GssaStoich* s = static_cast< GssaStoich* >( c->data() );
	// s->updateRates( v, dt );
}

void GssaStoich::rebuildMatrix( Eref stoich, vector< Id >& ret )
{
	Stoich::rebuildMatrix( stoich, ret );
	// Stuff here to set up the dependencies.
	unsigned int numRates = N_.nColumns();
	assert ( numRates = rates_.size() );
	transN_.setSize( numRates, N_.nRows() );
	N_.transpose( transN_ );
	dependency_.resize( numRates );
	for ( unsigned int i = 0; i < numRates; ++i ) {
		transN_.getGillespieDependence( i, dependency_[ i ] );
/*

		vector< unsigned int > depIndex;
		vector< RateTerm* > deps;
		transN_.getGillespieDependence( i, depIndex );
		for ( vector< unsigned int >::iterator j = depIndex.begin();
			j != depIndex.end(); ++j ) {
			assert( *j < numRates );
			deps.push_back( rates_[ *j ] );
		}
		dependency_.push_back( deps );
*/
	}
}

unsigned int GssaStoich::pickReac()
{
	double r = mtrand() * atot_;
	double sum = 0.0;
	// This is an inefficient way to do it. Can easily get to 
	// log time or thereabouts by doing one or two levels of 
	// subsidiary tables. Slepoy, Thompson and Plimpton 2008
	// report a linear time version.
	for ( vector< double >::iterator i = v_.begin(); i != v_.end(); ++i )
		if ( r < ( sum += *i ) )
			return static_cast< unsigned int >( i - v_.begin() );
	return v_.size();
}

void GssaStoich::innerProcessFunc( Eref e, ProcInfo info )
{
	double t = info->currTime_;
	double nextt = t + info->dt_;
	while ( t < nextt ) {
		// Figure out when the reaction will occur. The atot_
		// calculation actually estimates time for which reaction will
		// NOT occur, as atot_ sums all propensities.
		if ( atot_ <= 0.0 ) // Nothing is going to happen.
			break;
		double dt = ( 1.0 / atot_ ) * log( 1.0 / mtrand() );
		t += dt;
		if ( t >= nextt ) { // bail out if we run out of time.
			// We save the t and rindex past the checkpoint, so
			// as to continue if needed. However, checkpoint
			// may also involve changes to rates, in which
			// case these values may be invalidated. I worry
			// about an error here.
			continuationT_ = t;
			// continuationReac_ = rindex;
			break;
		}
		unsigned int rindex = pickReac(); // Does the first randnum call
		if ( rindex == rates_.size() ) 
			break;
		transN_.fireReac( rindex, S_ );
		updateDependentRates( dependency_[ rindex ] );
	}
}

void GssaStoich::updateDependentRates( const vector< unsigned int >& deps )
{
	for( vector< unsigned int >::const_iterator i = deps.begin(); 
		i != deps.end(); ++i ) {
		atot_ -= v_[ *i ];
		atot_ += ( v_[ *i ] = ( *rates_[ *i ] )() );
	}
}
