/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include <math.h>
#include "moose.h"
#include "Nernst.h"

const double Nernst::R_OVER_F = 8.6171458e-5;
const double Nernst::ZERO_CELSIUS = 273.15;

const Cinfo* initNernstCinfo()
{

	///////////////////////////////////////////////////////
	// Field definitions
	///////////////////////////////////////////////////////
	static Finfo* NernstFinfos[] =
	{
		new ValueFinfo( "E", ValueFtype1< double >::global(),
			GFCAST( &Nernst::getE ),
			&dummyFunc
		),
		new ValueFinfo( "Temperature", ValueFtype1< double >::global(),
			GFCAST( &Nernst::getTemperature ), 
			RFCAST( &Nernst::setTemperature )
		),
		new ValueFinfo( "valence", ValueFtype1< int >::global(),
			GFCAST( &Nernst::getValence ), 
			RFCAST( &Nernst::setValence )
		),
		new ValueFinfo( "Cin", ValueFtype1< double >::global(),
			GFCAST( &Nernst::getCin ), 
			RFCAST( &Nernst::setCin )
		),
		new ValueFinfo( "Cout", ValueFtype1< double >::global(),
			GFCAST( &Nernst::getCout ), 
			RFCAST( &Nernst::setCout )
		),
		new ValueFinfo( "scale", ValueFtype1< double >::global(),
			GFCAST( &Nernst::getScale ), 
			RFCAST( &Nernst::setScale )
		),
	///////////////////////////////////////////////////////
	// Shared definitions
	///////////////////////////////////////////////////////

	///////////////////////////////////////////////////////
	// MsgSrc definitions
	///////////////////////////////////////////////////////
		new SrcFinfo( "ESrc", Ftype1< double >::global() ),

	///////////////////////////////////////////////////////
	// MsgDest definitions
	///////////////////////////////////////////////////////
		new DestFinfo( "CinMsg", Ftype1< double >::global(),
			RFCAST( &Nernst::cinFunc ) ),
		new DestFinfo( "CoutMsg", Ftype1< double >::global(),
			RFCAST( &Nernst::coutFunc ) ),

	};

	static string doc[] =
	{
		"Name", "Nernst",
		"Author", "Upinder S. Bhalla, 2007, NCBS",
		"Description", "Nernst: Calculates Nernst potential for a given ion based on "
				"Cin and Cout, the inside and outside concentrations. "
				"Immediately sends out the potential to all targets.",
	};
	
	static const Cinfo NernstCinfo(
		doc,
		sizeof( doc ) / sizeof( string ),		
		initNeutralCinfo(),
		NernstFinfos,
		sizeof( NernstFinfos ) / sizeof(Finfo *),
		ValueFtype1< Nernst >::global()
	);

	return &NernstCinfo;
}

static const Cinfo* nernstCinfo = initNernstCinfo();
static const Slot eSrcSlot =
	initNernstCinfo()->getSlot( "ESrc" );

///////////////////////////////////////////////////
// Field function definitions
///////////////////////////////////////////////////

double Nernst::getE( Eref e ) {
	static_cast< Nernst* >( e.data() )->updateE();
	return static_cast< Nernst* >( e.data() )->E_;
}

void Nernst::localSetTemperature( double value ) {
	if ( value > 0.0 ) {
		Temperature_ = value;
		factor_ = scale_ * R_OVER_F * Temperature_ / valence_;
	}
}
void Nernst::setTemperature( const Conn* c, double value ) {
	static_cast< Nernst* >( c->data() )->localSetTemperature( value );
}
double Nernst::getTemperature( Eref e ) {
	return static_cast< Nernst* >( e.data() )->Temperature_;
}

void Nernst::localSetValence( int value ) {
	if ( value != 0 ) {
		valence_ = value;
		factor_ = scale_ * R_OVER_F * Temperature_ / valence_;
	}
}
void Nernst::setValence( const Conn* c, int value ) {
	static_cast< Nernst* >( c->data() )->localSetValence( value );
}
int Nernst::getValence( Eref e ) {
	return static_cast< Nernst* >( e.data() )->valence_;
}

void Nernst::setCin( const Conn* c, double value ) {
	static_cast< Nernst* >( c->data() )->Cin_ = value;
}
double Nernst::getCin( Eref e ) {
	return static_cast< const Nernst* >( e.data() )->Cin_;
}

void Nernst::setCout( const Conn* c, double value ) {
	static_cast< Nernst* >( c->data() )->Cout_ = value;
}
double Nernst::getCout( Eref e ) {
	return static_cast< const Nernst* >( e.data() )->Cout_;
}

void Nernst::setScale( const Conn* c, double value ) {
	static_cast< Nernst* >( c->data() )->scale_ = value;
}
double Nernst::getScale( Eref e ) {
	return static_cast< const Nernst* >( e.data() )->scale_;
}

///////////////////////////////////////////////////
// Dest function definitions
///////////////////////////////////////////////////

void Nernst::updateE( )
{
	E_ = factor_ * log( Cout_ / Cin_ );
}

void Nernst::cinFuncLocal( const Conn* c, double conc )
{
	Cin_ = conc;
	updateE();
	send1< double >( c->target(), eSrcSlot, E_ );
}
void Nernst::cinFunc( const Conn* c, double value ) {
	static_cast< Nernst* >( c->data() )->cinFuncLocal( c, value );
}

void Nernst::coutFuncLocal( const Conn* c, double conc )
{
	Cout_ = conc;
	updateE();
	send1< double >( c->target(), eSrcSlot, E_ );
}
void Nernst::coutFunc( const Conn* c, double value ) {
	static_cast< Nernst* >( c->data() )->coutFuncLocal( c, value );
}


///////////////////////////////////////////////////
// Unit tests
///////////////////////////////////////////////////

#ifdef DO_UNIT_TESTS
#include "../element/Neutral.h"
void testNernst()
{
	cout << "\nTesting Nernst" << flush;

	Element* n = Neutral::create( "Neutral", "n", Element::root()->id(), 
		Id::scratchId() );
	Element* nernst = Neutral::create( "Nernst", "Ca", n->id(), 
		Id::scratchId() );
	SetConn c( nernst, 0 );
	ASSERT( nernst != 0, "creating Nernst" );
	Nernst::setValence( &c, 1 );
	Nernst::setCin( &c, 0.01 );
	Nernst::setCout( &c, 1.0 );
	double E = Nernst::getE( nernst );
	ASSERT( fabs( E - 0.0585 * 2.0 ) < 0.001, "Testing Nernst" );

	// Get rid of the test stuff
	set( n, "destroy" );
}
#endif 
