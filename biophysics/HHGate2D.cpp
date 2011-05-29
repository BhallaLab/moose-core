/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
#include "ElementValueFinfo.h"
#include "../builtins/Interpol2D.h"
#include "HHGate2D.h"

static const double SINGULARITY = 1.0e-6;

const Cinfo* HHGate2D::initCinfo()
{
	///////////////////////////////////////////////////////
	// Field definitions.
	///////////////////////////////////////////////////////
		static ReadOnlyLookupValueFinfo< HHGate2D, vector< double >, double >
			A( "A",
			"lookupA: Look up the A gate value from two doubles, passed"
			"in as a vector. Uses linear interpolation in the 2D table"
			 "The range of the lookup doubles is predefined based on "
			 "knowledge of voltage or conc ranges, and the granularity "
			 "is specified by the xmin, xmax, and dx field, and their "
			 "y-axis counterparts.",
			&HHGate2D::lookupA );
		static ReadOnlyLookupValueFinfo< HHGate2D, vector< double >, double >
			B( "B",
			"lookupB: Look up B gate value from two doubles in a vector.",
			&HHGate2D::lookupB );

		static FieldElementFinfo< HHGate2D, Interpol2D > tableA( 
			"tableA",
			"Table of A entries",
			Interpol2D::initCinfo(),
			&HHGate2D::getTableA,
			&HHGate2D::setNumTable,
			&HHGate2D::getNumTable
		);

		static FieldElementFinfo< HHGate2D, Interpol2D > tableB( 
			"tableB",
			"Table of B entries",
			Interpol2D::initCinfo(),
			&HHGate2D::getTableB,
			&HHGate2D::setNumTable,
			&HHGate2D::getNumTable
		);
	///////////////////////////////////////////////////////
	// DestFinfos
	///////////////////////////////////////////////////////
	static Finfo* HHGate2DFinfos[] =
	{
		&A,			// ReadOnlyLookupValue
		&B,			// ReadOnlyLookupValue
		&tableA,	// ElementValue
		&tableB,	// ElementValue
	};

	static string doc[] =
	{
		"Name", "HHGate2D",
		"Author", "Niraj Dudani, 2009, NCBS",
		"Description", "HHGate2D: Gate for Hodkgin-Huxley type channels, equivalent to the "
		"m and h terms on the Na squid channel and the n term on K. "
		"This takes the voltage and state variable from the channel, "
		"computes the new value of the state variable and a scaling, "
		"depending on gate power, for the conductance. These two "
		"terms are sent right back in a message to the channel.",
	};

	static Cinfo HHGate2DCinfo(
		"HHGate2D",
		Neutral::initCinfo(),
		HHGate2DFinfos, sizeof(HHGate2DFinfos)/sizeof(Finfo *),
		new Dinfo< HHGate2D >()
	);

	return &HHGate2DCinfo;
}

static const Cinfo* hhGate2DCinfo = HHGate2D::initCinfo();
///////////////////////////////////////////////////
HHGate2D::HHGate2D()
	: originalChanId_(0),
		originalGateId_(0)
{;}

HHGate2D::HHGate2D( Id originalChanId, Id originalGateId )
	: 
		originalChanId_( originalChanId ),
		originalGateId_( originalGateId )
{;}

///////////////////////////////////////////////////
// Field function definitions
///////////////////////////////////////////////////
double HHGate2D::lookupA( vector< double > v ) const
{
	if ( v.size() < 2 ) {
		cerr << "Error: HHGate2D::getAValue: 2 real numbers needed to lookup 2D table.\n";
		return 0.0;
	}
	
	if ( v.size() > 2 ) {
		cerr << "Error: HHGate2D::getAValue: Only 2 real numbers needed to lookup 2D table. "
			"Using only first 2.\n";
	}
	
	return A_.innerLookup( v[ 0 ], v[ 1 ] );
}

double HHGate2D::lookupB( vector< double > v ) const
{
	if ( v.size() < 2 ) {
		cerr << "Error: HHGate2D::getAValue: 2 real numbers needed to lookup 2D table.\n";
		return 0.0;
	}
	
	if ( v.size() > 2 ) {
		cerr << "Error: HHGate2D::getAValue: Only 2 real numbers needed to lookup 2D table. "
			"Using only first 2.\n";
	}
	
	return B_.innerLookup( v[ 0 ], v[ 1 ] );
}

void HHGate2D::lookupBoth( double v, double c, double* A, double* B ) const
{
	*A = A_.innerLookup( v, c );
	*B = B_.innerLookup( v, c );
}


///////////////////////////////////////////////////
// Access functions for Interpols
///////////////////////////////////////////////////

Interpol2D* HHGate2D::getTableA( unsigned int i )
{
	return &A_;
}

Interpol2D* HHGate2D::getTableB( unsigned int i )
{
	return &B_;
}

unsigned int HHGate2D::getNumTable() const
{
	return 1;
}

void HHGate2D::setNumTable( unsigned int i)
{
	;
}


///////////////////////////////////////////////////
// Functions to check if this is original or copy
///////////////////////////////////////////////////
bool HHGate2D::isOriginalChannel( Id id ) const
{
	return ( id == originalChanId_ );
}

bool HHGate2D::isOriginalGate( Id id ) const
{
	return ( id == originalGateId_ );
}

Id HHGate2D::originalChannelId() const
{
	return originalChanId_;
}
///////////////////////////////////////////////////
// Dest function definitions
///////////////////////////////////////////////////
/*
void HHGate2D::gateFunc( const Eref& e, const Qinfo* q,
	double v1, double v2 )
{

	sendBack2< double, double >( c, gateSlot,
		h->A_.innerLookup( v1, v2 ) , h->B_.innerLookup( v1, v2 ) );
}
*/

/**
 * Request the gate explicitly to create Interpols, with the given ids. This is
 * used when the gate is a global object, and so the interpols need to be
 * globals too. Comes in use in TABCREATE in the parallel context.
void HHGate2D::createInterpols( const Conn* c, IdGenerator idGen )
{
	HHGate2D* h = static_cast< HHGate2D *>( c->data() );
	Eref e = c->target();
	
	const Cinfo* ic = initInterpol2DCinfo();
	
	// Here we must set the noDelFlag to 1 because these data
	// parts belong to the parent HHGate2D structure.
	Element* A = ic->create( 
		idGen.next(), "A", static_cast< void* >( &h->A_ ), 1 );
	e.add( "childSrc", A, "child" );

	Element* B = ic->create( 
		idGen.next(), "B", static_cast< void* >( &h->B_), 1 );
	e.add( "childSrc", B, "child" );
}
 */
