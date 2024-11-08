/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "../basecode/header.h"
#include "../basecode/ElementValueFinfo.h"
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

		static ElementValueFinfo< HHGate2D, vector< vector< double > > > tableA(
			"tableA",
			"Table of A entries",
                        &HHGate2D::setTableA,
			&HHGate2D::getTableA
		);

		static ElementValueFinfo< HHGate2D, vector< vector< double > > > tableB(
			"tableB",
			"Table of B entries",
			&HHGate2D::setTableB,
			&HHGate2D::getTableB);

		static ElementValueFinfo< HHGate2D, double > xmin( "xmin",
			"Minimum range for lookup",
			&HHGate2D::setXmin,
			&HHGate2D::getXmin
		);

		static ElementValueFinfo< HHGate2D, double > xmax( "xmax",
			"Minimum range for lookup",
			&HHGate2D::setXmax,
			&HHGate2D::getXmax
		);

		static ElementValueFinfo< HHGate2D, unsigned int > xdivs( "xdivs",
			"Divisions for lookup. Zero means to use linear interpolation",
			&HHGate2D::setXdivs,
			&HHGate2D::getXdivs);

		static ElementValueFinfo< HHGate2D, double > ymin( "ymin",
			"Minimum range for lookup",
			&HHGate2D::setYmin,
			&HHGate2D::getYmin);

		static ElementValueFinfo< HHGate2D, double > ymax( "ymax",
			"Minimum range for lookup",
			&HHGate2D::setYmax,
			&HHGate2D::getYmax);

		static ElementValueFinfo< HHGate2D, unsigned int > ydivs( "ydivs",
			"Divisions for lookup. Zero means to use linear interpolation",
			&HHGate2D::setYdivs,
			&HHGate2D::getYdivs);

        ///////////////////////////////////////////////////////
	// DestFinfos
	///////////////////////////////////////////////////////
	static Finfo* HHGate2DFinfos[] =
	{
		&A,			// ReadOnlyLookupValue
		&B,			// ReadOnlyLookupValue
		&tableA,	// ElementValue
		&tableB,	// ElementValue
                &xmin,
                &xmax,
                &xdivs,
                &ymin,
                &ymax,
                &ydivs,
        };

	static string doc[] =
	{
		"Name", "HHGate2D",
		"Author", "Niraj Dudani, 2009, NCBS. Updated by Subhasis Ray, 2014, 2024 NCBS.",
		"Description", "HHGate2D: Gate for Hodkgin-Huxley type channels, equivalent to the "
		"m and h terms on the Na squid channel and the n term on K. "
		"This takes the voltage and state variable from the channel, "
		"computes the new value of the state variable and a scaling, "
		"depending on gate power, for the conductance. These two "
		"terms are sent right back in a message to the channel.",
	};

        static Dinfo< HHGate2D > dinfo;
	static Cinfo HHGate2DCinfo(
		"HHGate2D",
		Neutral::initCinfo(),
		HHGate2DFinfos, sizeof(HHGate2DFinfos)/sizeof(Finfo *),
                &dinfo,
                doc,
                sizeof(doc) / sizeof(string)
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

vector< vector< double > > HHGate2D::getTableA( const Eref& e ) const
{
    return A_.getTableVector();
}

void HHGate2D::setTableA(const Eref& e, vector< vector< double > > value )
{
    A_.setTableVector(value);
}

vector< vector< double > > HHGate2D::getTableB(const Eref& e) const
{
    return B_.getTableVector();
}

void HHGate2D::setTableB(const Eref& e, vector< vector< double > > value )
{
    B_.setTableVector(value);
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

double HHGate2D::getXmin(const Eref& e) const
{
    return A_.getXmin();
}

void HHGate2D::setXmin(const Eref& e, double value)
{
    A_.setXmin(value);
    B_.setXmin(value);
}

double HHGate2D::getXmax(const Eref& e) const
{
    return A_.getXmax();
}

void HHGate2D::setXmax(const Eref& e, double value)
{
    A_.setXmax(value);
    B_.setXmax(value);
}

unsigned int HHGate2D::getXdivs(const Eref& e) const
{
    return A_.getXdivs();
}

void HHGate2D::setXdivs(const Eref& e, unsigned int value)
{
    A_.setXdivs(value);
    B_.setXdivs(value);
}

double HHGate2D::getYmin(const Eref& e) const
{
    return A_.getYmin();
}

void HHGate2D::setYmin(const Eref& e, double value)
{
    A_.setYmin(value);
    B_.setYmin(value);
}

double HHGate2D::getYmax(const Eref& e) const
{
    return A_.getYmax();
}

void HHGate2D::setYmax(const Eref& e, double value)
{
    A_.setYmax(value);
    B_.setYmax(value);
}

unsigned int HHGate2D::getYdivs(const Eref& e) const
{
    return A_.getYdivs();
}

void HHGate2D::setYdivs(const Eref& e, unsigned int value)
{
    A_.setYdivs(value);
    B_.setYdivs(value);
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
