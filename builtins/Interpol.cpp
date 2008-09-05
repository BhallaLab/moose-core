/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include <fstream>
#include <math.h>
#include "moose.h"
#include "Interpol.h"

const double Interpol::EPSILON = 1.0e-10;
const unsigned int Interpol::MAX_DIVS = 10000000;
	//Ten million points should do.

const Cinfo* initInterpolCinfo()
{
	/**
	 * This is a shared message for doing lookups on the table.
	 * Receives a double with the x value
	 * Sends back a double with the looked-up y value.
	 */
	static Finfo* lookupReturnShared[] =
	{
		new DestFinfo( "lookup", Ftype1< double >::global(),
						RFCAST( &Interpol::lookupReturn ) ),
		new SrcFinfo( "trig", Ftype1< double >::global() ),
	};
	static Finfo* interpolFinfos[] =
	{
	///////////////////////////////////////////////////////
	// Field definitions
	///////////////////////////////////////////////////////
		new ValueFinfo( "xmin", ValueFtype1< double >::global(),
			GFCAST( &Interpol::getXmin ),
			RFCAST( &Interpol::setXmin )
		),
		new ValueFinfo( "xmax", ValueFtype1< double >::global(),
			GFCAST( &Interpol::getXmax ),
			RFCAST( &Interpol::setXmax )
		),
		new ValueFinfo( "xdivs", ValueFtype1< int >::global(),
			GFCAST( &Interpol::getXdivs ),
			RFCAST( &Interpol::setXdivs )
		),
		new ValueFinfo( "mode", ValueFtype1< int >::global(),
			GFCAST( &Interpol::getMode ),
			RFCAST( &Interpol::setMode )
		),
		new ValueFinfo( "calc_mode", ValueFtype1< int >::global(),
			GFCAST( &Interpol::getMode ),
			RFCAST( &Interpol::setMode )
		),
		new ValueFinfo( "dx", ValueFtype1< double >::global(),
			GFCAST( &Interpol::getDx ),
			RFCAST( &Interpol::setDx )
		),
		new ValueFinfo( "sy", ValueFtype1< double >::global(),
			GFCAST( &Interpol::getSy ),
			RFCAST( &Interpol::setSy )
		),
		new LookupFinfo( "table",
			LookupFtype< double, unsigned int >::global(),
			GFCAST( &Interpol::getTable ),
			RFCAST( &Interpol::setTable )
		),
		new ValueFinfo( "tableVector", ValueFtype1< vector< double > >::global(),
			GFCAST( &Interpol::getTableVector ),
			RFCAST( &Interpol::setTableVector )
		),
	///////////////////////////////////////////////////////
	// Shared message definitions
	///////////////////////////////////////////////////////
		new SharedFinfo( "lookupReturn", lookupReturnShared, 2 ),
		
	///////////////////////////////////////////////////////
	// MsgSrc definitions
	///////////////////////////////////////////////////////
		new SrcFinfo( "lookupSrc", Ftype1< double >::global() ),

	///////////////////////////////////////////////////////
	// MsgDest definitions
	///////////////////////////////////////////////////////
		/// Look up and interpolate value from table using x value.
		new DestFinfo( "lookup", Ftype1< double >::global(), 
			RFCAST( &Interpol::lookup )
		),

		/// Expand table. First arg is xdivs, second is mode.
		new DestFinfo( "tabFill", Ftype2< int, int >::global(), 
			RFCAST( &Interpol::tabFill )
		),
		/// Print contents to file.
		new DestFinfo( "print", Ftype1< string >::global(), 
			RFCAST( &Interpol::print )
		),
		/// Append contents to file.
		new DestFinfo( "append", Ftype1< string >::global(), 
			RFCAST( &Interpol::append )
		),
		/// Load contents from file.
		//  load filename skiplines
		new DestFinfo( "load", Ftype2< string, unsigned int >::global(), 
			RFCAST( &Interpol::load )
		),
		/// Append to existing table. Used in loadtab -continue.
		new DestFinfo( "appendTableVector", 
			Ftype1< vector< double > >::global(), 
			RFCAST( &Interpol::appendTableVector )
		),
	};

	static Cinfo interpolCinfo(
	"Interpol",
	"Upinder S. Bhalla, 2007, NCBS",
	"Interpol: Interpolation class. Handles lookup of a y value from an\nx value, where the x value is a double. Can either use\ninterpolation or roundoff to the nearest index.",
	initNeutralCinfo(),
	interpolFinfos,
	sizeof( interpolFinfos ) / sizeof( Finfo * ),
	ValueFtype1< Interpol >::global()
	);

	return &interpolCinfo;
}

static const Cinfo* interpolCinfo = initInterpolCinfo();

static const Slot lookupReturnSlot = 
	initInterpolCinfo()->getSlot( "lookupReturn.trig" );
static const Slot lookupSlot = initInterpolCinfo()->getSlot( "lookupSrc" );

////////////////////////////////////////////////////////////////////
// Here we set up Interpol value fields
////////////////////////////////////////////////////////////////////

void Interpol::setXmin( const Conn* c, double xmin ) 
{
	static_cast< Interpol* >( c->data() )->localSetXmin( xmin );
}
double Interpol::getXmin( Eref e )
{
	return static_cast< Interpol* >( e.data() )->xmin_;
}

void Interpol::setXmax( const Conn* c, double xmax ) 
{
	static_cast< Interpol* >( c->data() )->localSetXmax( xmax );
}
double Interpol::getXmax( Eref e )
{
	return static_cast< Interpol* >( e.data() )->xmax_;
}

void Interpol::setXdivs( const Conn* c, int xdivs ) 
{
	static_cast< Interpol* >( c->data() )->localSetXdivs( xdivs );
}
int Interpol::getXdivs( Eref e )
{
	return static_cast< Interpol* >( e.data() )->table_.size() - 1;
}

void Interpol::setDx( const Conn* c, double dx ) 
{
	static_cast< Interpol* >( c->data() )->localSetDx( dx );
}
double Interpol::getDx( Eref e )
{
	return static_cast< Interpol* >( e.data() )->localGetDx();
}

void Interpol::setSy( const Conn* c, double value ) 
{
	static_cast< Interpol* >( c->data() )->localSetSy( value );
}
double Interpol::getSy( Eref e )
{
	return static_cast< Interpol* >( e.data() )->sy_;
}

void Interpol::setMode( const Conn* c, int value ) 
{
	static_cast< Interpol* >( c->data() )->mode_ = value;
}
int Interpol::getMode( Eref e )
{
	return static_cast< Interpol* >( e.data() )->mode_;
}

void Interpol::setTable(
				const Conn* c, double val, const unsigned int& i )
{
	static_cast< Interpol* >( c->data() )->setTableValue( val, i );
}
double Interpol::getTable( Eref e, const unsigned int& i )
{
	return static_cast< Interpol* >( e.data() )->getTableValue( i );
}

void Interpol::setTableVector( const Conn* c, vector< double > value ) 
{
	static_cast< Interpol* >( c->data() )->localSetTableVector( value );
}

vector< double > Interpol::getTableVector( Eref e )
{
	return static_cast< Interpol* >( e.data() )->table_;
}

////////////////////////////////////////////////////////////////////
// Here we set up Interpol Destination functions
////////////////////////////////////////////////////////////////////

/**
 * lookupReturn uses its argument to do an interpolating lookup of the
 * table. It sends a return message to the
 * originating object with the looked up value.
 */
void Interpol::lookupReturn( const Conn* c, double val )
{
	double ret =
			static_cast< Interpol* >( c->data() )->innerLookup( val );
	sendBack1< double >( c, lookupReturnSlot, ret );
}

/**
 * lookup uses its argument to do an interpolating lookup of the
 * table. It sends a message with this looked-up value
 * on to any targets using lookupSrc
 */
void Interpol::lookup( const Conn* c, double val )
{
	double ret =
			static_cast< Interpol* >( c->data() )->innerLookup( val );
	send1< double >( c->target(), lookupSlot, ret );
}

void Interpol::tabFill( const Conn* c, int xdivs, int mode )
{
	static_cast< Interpol* >( c->data() )->innerTabFill( xdivs, mode );
}

void Interpol::print( const Conn* c, string fname )
{
	static_cast< Interpol* >( c->data() )->innerPrint( fname, 0 );
}

void Interpol::append( const Conn* c, string fname )
{
	static_cast< Interpol* >( c->data() )->innerPrint( fname, 1 );
}

void Interpol::load( const Conn* c, string fname, unsigned int skiplines )
{
	static_cast< Interpol* >( c->data() )->innerLoad( fname, skiplines );
}

void Interpol::appendTableVector( const Conn* c, vector< double > value ) 
{
	static_cast< Interpol* >( c->data() )->localAppendTableVector( value );
}

////////////////////////////////////////////////////////////////////
// Here we set up private Interpol class functions.
////////////////////////////////////////////////////////////////////


Interpol::Interpol( unsigned long xdivs, double xmin, double xmax )
	: xmin_( xmin ), xmax_( xmax ), sy_( 1.0 )
{
	table_.resize( xdivs + 1, 0.0 );
	mode_ = 1; // Mode 1 is linear interpolation. 0 is indexing.
	if ( fabs( xmax_ - xmin_ ) > EPSILON )
		invDx_ = static_cast< double >( table_.size() ) /
			( xmax_ - xmin_);
	else 
		invDx_ = 1.0;
}

double Interpol::interpolateWithoutCheck( double x ) const
{
	double xv = ( x - xmin_ ) * invDx_;
	unsigned long ixv = static_cast< unsigned long >( xv );
	assert( ixv < table_.size() - 1 );
	// return table_[ i ] + ( table_[ i + 1 ] - table_ [ i ] ) * ( xv - i );
	vector< double >::const_iterator i = table_.begin() + ixv;
	return *i + ( *( i + 1 ) - *i ) * ( xv - ixv ); 

	/*
	vector< double >::const_iterator i = table_.begin() + 
		static_cast< unsigned long >( xv );
	return *i + ( *( i + 1 ) - *i ) * ( xv - floor( xv ) ); 
	*/
}

double Interpol::innerLookup( double x ) const
{
	if ( table_.size() == 0 )
		return 0.0;
	if ( x <= xmin_ ) 
		return table_.front();
	if ( x >= xmax_ )
		return table_.back();
	if ( mode_ == 0 )
		return indexWithoutCheck( x );
	else 
		return interpolateWithoutCheck( x );
}

bool Interpol::operator==( const Interpol& other ) const
{
	return (
		xmin_ == other.xmin_ &&
		xmax_ == other.xmax_ &&
		mode_ == other.mode_ &&
		table_ == other.table_ );
}

bool Interpol::operator<( const Interpol& other ) const
{
	if ( *this == other )
		return 0;
	if ( table_.size() < other.table_.size() )
		return 1;
	if ( table_.size() > other.table_.size() )
		return 0;
	for (size_t i = 0; i < table_.size(); i++) {
		if ( table_[i] < other.table_[i] )
			return 1;
		if ( table_[i] > other.table_[i] )
			return 0;
	}
	return 0;
}

void Interpol::localSetXmin( double value ) {
	if ( fabs( xmax_ - value) > EPSILON ) {
		xmin_ = value;
		invDx_ = static_cast< double >( table_.size() - 1 ) / 
			( xmax_ - xmin_ );
	} else {
		cerr << "Warning: Interpol: Xmin ~= Xmax : Assignment failed\n";
	}
}
void Interpol::localSetXmax( double value ) {
	if ( fabs( value - xmin_ ) > EPSILON ) {
		xmax_ = value;
		invDx_ = static_cast< double >( table_.size() - 1 ) / 
			( xmax_ - xmin_ );
	} else {
		cerr << "Warning: Interpol: Xmin ~= Xmax : Assignment failed\n";
	}
}
void Interpol::localSetXdivs( int value ) {
	if ( value > 0 ) {
		table_.resize( value + 1 );
		invDx_ = static_cast< double >( value ) / ( xmax_ - xmin_ );
	}
}
// Later do interpolation etc to preseve contents.
// Later also check that it is OK for xmax_ < xmin_
void Interpol::localSetDx( double value ) {
	if ( fabs( value ) - EPSILON > 0 ) {
		unsigned int xdivs = static_cast< unsigned int >( 
			0.5 + fabs( xmax_ - xmin_ ) / value );
		if ( xdivs < 1 || xdivs > MAX_DIVS ) {
			cerr << "Warning: Interpol: Out of range:" <<
				xdivs << " entries in table.\n";
				return;
		}
		table_.resize( xdivs + 1 );
		invDx_ = static_cast< double >( xdivs ) / 
			( xmax_ - xmin_ );
	}
}
double Interpol::localGetDx() const {
	return ( xmax_ - xmin_ ) / static_cast< double >( table_.size() - 1 );
}

void Interpol::localSetSy( double value ) {
	if ( fabs( value ) - EPSILON > 0 ) {
		double ratio = value / sy_;
		vector< double >::iterator i;
		for ( i = table_.begin(); i != table_.end(); i++) 
			*i *= ratio;
		sy_ = value;
	} else {
		cerr << "Warning: Interpol: localSetSy: sy too small:" <<
			value << "\n";
	}
}

void Interpol::setTableValue( double value, unsigned int index ) {
	if ( index < table_.size() )
		table_[ index ] = value;
}

double Interpol::getTableValue( unsigned int index ) const {
	if ( index < table_.size() )
		return table_[ index ];
	return 0.0;
}

// This sets the whole thing up: values, xdivs, dx and so on. Only xmin
// and xmax are unknown to the input vector.
void Interpol::localSetTableVector( const vector< double >& value ) 
{
	unsigned int xdivs = value.size();
	if ( xdivs == 0 ) {
		table_.resize( 0 );
		invDx_ = 1.0;
	} else {
		invDx_ = static_cast< double >( xdivs ) / ( xmax_ - xmin_ );
		table_ = value;
	}
}

// This sets the whole thing up: values, xdivs, dx and so on. Only xmin
// and xmax are unknown to the input vector.
void Interpol::localAppendTableVector( const vector< double >& value ) 
{
	unsigned int xdivs = value.size();
	if ( xdivs == 0 ) {
		return;
	} else {
		xdivs += table_.size();
		invDx_ = static_cast< double >( xdivs ) / ( xmax_ - xmin_ );
		table_.insert( table_.end(), value.begin(), value.end() );
	}
}

void Interpol::innerTabFill( int xdivs, int mode )
{
	int oldMode = mode_;
	vector< double > newtab;
	newtab.resize( xdivs + 1, 0.0 );
	double dx = ( xmax_ - xmin_ ) / static_cast< double >( xdivs );
	mode_ = 1; // Has to be, so we can interpolate.
	for ( int i = 0; i <= xdivs; i++ )
		newtab[ i ] = innerLookup(
			xmin_ + dx * static_cast< double >( i ) );
	table_ = newtab;
	mode_ = oldMode;
	invDx_ = 1.0/dx;
}

void Interpol::innerPrint( const string& fname, bool appendFlag )
{
	vector< double >::iterator i;
	if ( appendFlag ) {
		std::ofstream fout( fname.c_str(), std::ios::app );
		for ( i = table_.begin(); i != table_.end(); i++ )
			fout << *i << endl;
	} else {
		std::ofstream fout( fname.c_str(), std::ios::trunc );
		for ( i = table_.begin(); i != table_.end(); i++ )
			fout << *i << endl;
	}
}

void Interpol::innerLoad( const string& fname, unsigned int skiplines )
{
	vector< double >::iterator i;
	std::ifstream fin( fname.c_str() );
	string line;
	if ( fin.good() ) {
		unsigned int i;
		for ( i = 0; i < skiplines; i++ ) {
			if ( fin.good () )
				getline( fin, line );
			else
				break;
		}
		if ( !fin.good() )
			return;
		double y;
		table_.resize( 0 );
		while ( fin >> y ) 
			table_.push_back( y );
	} else {
		cout << "Error: Interpol::innerLoad: Failed to open file " << 
			fname << endl;
	}
}

#ifdef DO_UNIT_TESTS
void testInterpol()
{
	static const unsigned int XDIVS = 100;
	cout << "\nDoing Interpol tests";

	Element* i1 = interpolCinfo->create( Id::scratchId(), "i1" );
	Element* i2 = interpolCinfo->create( Id::scratchId(), "i2" );
	unsigned int i;
	double ret = 0;

	set< int >( i1, "xdivs", XDIVS );

	set< double >( i1, "xmin", 0 );
	get< double >( i1, "xmin", ret );
	ASSERT( ret == 0.0, "testInterpol" );

	set< double >( i1, "xmax", 20 );
	get< double >( i1, "xmax", ret );
	ASSERT( ret == 20.0, "testInterpol" );

	for ( i = 0; i <= XDIVS; i++ )
		lookupSet< double, unsigned int >(
						i1, "table", i * 10.0 - 100.0, i );

	for ( i = 0; i <= XDIVS; i++ ) {
		lookupGet< double, unsigned int >( i1, "table", ret, i );
		assert ( ret == i * 10.0 - 100.0 );
	}
	cout << ".";

	set< int >( i2, "xdivs", XDIVS );
	set< double >( i2, "xmin", 0 );
	set< double >( i2, "xmax", 10000.0 );

	// Here we use i2 as a dummy dest for the 
	// lookup operation, which takes place on i1.

	ASSERT(
		Eref( i1 ).add( "lookupSrc", i2, "xmin" ), "connecting interpols" );

		//	i1->findFinfo( "lookupSrc" )->add( i1, i2, i2->findFinfo( "xmin" ) ), "connecting interpols"

	set< double >( i1, "lookup", -10.0 );
	get< double >( i2, "xmin", ret );
	ASSERT( ret == -100.0, "Lookup minimum" );

	set< double >( i1, "lookup", 0.0 );
	get< double >( i2, "xmin", ret );
	ASSERT( ret == -100.0, "Lookup minimum" );

	set< double >( i1, "lookup", 2.0 );
	get< double >( i2, "xmin", ret );
	ASSERT( ret == 0.0, "Lookup middle" );

	set< double >( i1, "lookup", 2.1 );
	get< double >( i2, "xmin", ret );
	ASSERT( fabs( ret - 5.0 ) < 1.0e-10, "Lookup interpolation" );
	// ASSERT( ret == 5.0, "Lookup interpolation" );

	set< double >( i1, "lookup", 10.0 );
	get< double >( i2, "xmin", ret );
	ASSERT( ret == 400.0, "Lookup middle" );

	set< double >( i1, "lookup", 12.0 );
	get< double >( i2, "xmin", ret );
	ASSERT( ret == 500.0, "Lookup middle" );

	set< double >( i1, "lookup", 20.0 );
	get< double >( i2, "xmin", ret );
	ASSERT( ret == 900.0, "Lookup max" );

	set< double >( i1, "lookup", 20000.0 );
	get< double >( i2, "xmin", ret );
	ASSERT( ret == 900.0, "Lookup max" );
}

#endif // DO_UNIT_TESTS
