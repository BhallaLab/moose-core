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
#include "../element/Wildcard.h"
#include "RateTerm.h"
#include "KinSparseMatrix.h"
#include "Stoich.h"
#include "GssaStoich.h"
#include "randnum.h"

const Cinfo* initGssaStoichCinfo()
{
	static Finfo* processShared[] =
	{
		new DestFinfo( "process",
			Ftype1< ProcInfo >::global(),
			RFCAST( &GssaStoich::processFunc )),
		new DestFinfo( "reinit",
			Ftype1< ProcInfo >::global(),
			RFCAST( &GssaStoich::reinitFunc )),
	};

	static Finfo* process = new SharedFinfo( "process", processShared,
		sizeof( processShared ) / sizeof( Finfo* ) );

	/**
	 * Messages that connect to the GssaIntegrator object
	static Finfo* gssaShared[] =
	{
		new DestFinfo( "reinit", Ftype0::global(),
			&Stoich::reinitFunc ),
		new SrcFinfo( "assignStoich",
			Ftype1< void* >::global() ),
		new SrcFinfo( "assignY",
			Ftype2< double, unsigned int >::global() ),
	};
	 */
	//These are the fields of the stoich class
	static Finfo* gssaStoichFinfos[] =
	{
		///////////////////////////////////////////////////////
		// Field definitions
		///////////////////////////////////////////////////////
		new ValueFinfo( "method", 
			ValueFtype1< string >::global(),
			GFCAST( &GssaStoich::getMethod ), 
			RFCAST( &GssaStoich::setMethod )
		),
		new ValueFinfo( "path", 
			ValueFtype1< string >::global(),
			GFCAST( &GssaStoich::getPath ), 
			RFCAST( &GssaStoich::setPath )
		),
		///////////////////////////////////////////////////////
		// MsgSrc definitions
		///////////////////////////////////////////////////////
		///////////////////////////////////////////////////////
		// MsgDest definitions
		///////////////////////////////////////////////////////
		///////////////////////////////////////////////////////
		// Shared definitions
		///////////////////////////////////////////////////////
/*
		new SharedFinfo( "integrate", integrateShared, 
				sizeof( integrateShared )/ sizeof( Finfo* ) ),
		new SharedFinfo( "gssa", gssaShared, 
				sizeof( gssaShared )/ sizeof( Finfo* ) ),
*/
		process,
	};
	static SchedInfo schedInfo[] = { { process, 0, 0 } };

	static Cinfo gssaStoichCinfo(
		"GssaStoich",
		"Upinder S. Bhalla, 2008, NCBS",
		"GssaStoich: Gillespie Stochastic Simulation Algorithm object.\nClosely based on the Stoich object and inherits its \nhandling functions for constructing the matrix. Sets up stoichiometry matrix based calculations from a\nwildcard path for the reaction system.\nKnows how to compute derivatives for most common\nthings, also knows how to handle special cases where the\nobject will have to do its own computation. Generates a\nstoichiometry matrix, which is useful for lots of other\noperations as well.",
		initStoichCinfo(),
		gssaStoichFinfos,
		sizeof( gssaStoichFinfos )/sizeof(Finfo *),
		ValueFtype1< GssaStoich >::global(),
			schedInfo, 1
		);

	return &gssaStoichCinfo;
}

static const Cinfo* gssaStoichCinfo = initGssaStoichCinfo();

///////////////////////////////////////////////////
// Class function definitions
///////////////////////////////////////////////////

GssaStoich::GssaStoich()
	: Stoich(), atot_( 0.0 ), t_( 0.0 )
{
	useOneWayReacs_ = 1;
}
		
///////////////////////////////////////////////////
// Field function definitions
///////////////////////////////////////////////////

string GssaStoich::getMethod( Eref e )
{
	return static_cast< const GssaStoich* >( e.data() )->method_;
}
void GssaStoich::setMethod( const Conn* c, string method )
{
	static_cast< GssaStoich* >( c->data() )->innerSetMethod( method );
}

void GssaStoich::innerSetMethod( const string& method )
{
	method_ = method;
	cout << "in void GssaStoich::innerSetMethod( " << method << ") \n";
/*
	gssaMethod = G1;
	if ( method == "tauLeap" ) {
		gssaMethod = TauLeap;
	}
*/
}

string GssaStoich::getPath( Eref e ) {
	return static_cast< const GssaStoich* >( e.data() )->path_;
}

void GssaStoich::setPath( const Conn* c, string value ) {
	static_cast< GssaStoich* >( c->data() )->
	localSetPath( c->target(), value );
}

void GssaStoich::localSetPath( Eref stoich, const string& value )
{
	path_ = value;
	vector< Id > ret;
	wildcardFind( path_, ret );
	clear( stoich );
	if ( ret.size() > 0 ) {
		rebuildMatrix( stoich, ret );
	} else {
		cout << "No objects to simulate in path '" << value << "'\n";
	}
}

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
	s->t_ = 0.0;
	s->updateAllRates();
}

/*
// static func
void GssaStoich::integrateFunc( const Conn* c, vector< double >* v, double dt )
{
	// GssaStoich* s = static_cast< GssaStoich* >( c->data() );
	// s->updateRates( v, dt );
}
*/

void GssaStoich::rebuildMatrix( Eref stoich, vector< Id >& ret )
{
	Stoich::rebuildMatrix( stoich, ret );
	// Stuff here to set up the dependencies.
	unsigned int numRates = N_.nColumns();
	assert ( numRates == rates_.size() );
	transN_.setSize( numRates, N_.nRows() );
	N_.transpose( transN_ );
	dependency_.resize( numRates );
	for ( unsigned int i = 0; i < numRates; ++i ) {
		transN_.getGillespieDependence( i, dependency_[ i ] );
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

void GssaStoich::processFunc( const Conn* c, ProcInfo info )
{
	static_cast< GssaStoich* >( c->data() )->
		innerProcessFunc( c->target(), info );
}

void GssaStoich::innerProcessFunc( Eref e, ProcInfo info )
{
	double nextt = info->currTime_ + info->dt_;
	while ( t_ < nextt ) {
		// Figure out when the reaction will occur. The atot_
		// calculation actually estimates time for which reaction will
		// NOT occur, as atot_ sums all propensities.
		if ( atot_ <= 0.0 ) // Nothing is going to happen.
			break;
		if ( t_ > 0.0 ) {
			unsigned int rindex = pickReac(); // Does a randnum call
			if ( rindex == rates_.size() ) 
				break;
			transN_.fireReac( rindex, S_ );
			updateDependentRates( dependency_[ rindex ] );
		}
		double dt = ( 1.0 / atot_ ) * log( 1.0 / mtrand() );
		t_ += dt;
		if ( t_ >= nextt ) { // bail out if we run out of time.
			// We save the t past the checkpoint, so
			// as to continue if needed. However, checkpoint
			// may also involve changes to rates, in which
			// case these values may be invalidated. I worry
			// about an error here.
			break;
		}
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

void GssaStoich::updateAllRates()
{
	atot_ = 0.0;
	for( unsigned int i = 0; i < rates_.size(); ++i ) {
		atot_ += ( v_[ i ] = ( *rates_[ i ] )() );
	}
}
