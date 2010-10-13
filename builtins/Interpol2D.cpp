/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include <fstream>
#include <sstream>
#include "moose.h"
#include "Interpol.h"
#include "Interpol2D.h"

const Cinfo* initInterpol2DCinfo()
{
	static Finfo* lookupReturnShared[] =
	{
		new DestFinfo( "lookup", Ftype2< double, double >::global(),
						RFCAST( &Interpol2D::lookupReturn ) ),
		new SrcFinfo( "trig", Ftype1< double >::global() ),
	};
	
	static Finfo* interpol2DFinfos[] =
	{
	///////////////////////////////////////////////////////
	// Field definitions
	///////////////////////////////////////////////////////
		new ValueFinfo( "ymin", ValueFtype1< double >::global(),
			GFCAST( &Interpol2D::getYmin ),
			RFCAST( &Interpol2D::setYmin )
		),
		new ValueFinfo( "ymax", ValueFtype1< double >::global(),
			GFCAST( &Interpol2D::getYmax ),
			RFCAST( &Interpol2D::setYmax )
		),
		new ValueFinfo( "ydivs", ValueFtype1< int >::global(),
			GFCAST( &Interpol2D::getYdivs ),
			RFCAST( &Interpol2D::setYdivs )
		),
		new ValueFinfo( "dy", ValueFtype1< double >::global(),
			GFCAST( &Interpol2D::getDy ),
			RFCAST( &Interpol2D::setDy )
		),
		new LookupFinfo( "table",
			LookupFtype< double, vector< unsigned int > >::global(),
			GFCAST( &Interpol2D::getTable2D ),
			RFCAST( &Interpol2D::setTable2D )
		),		
		new LookupFinfo( "table2D",
			LookupFtype< double, vector< unsigned int > >::global(),
			GFCAST( &Interpol2D::getTable2D ),
			RFCAST( &Interpol2D::setTable2D )
		),
		new ValueFinfo( "tableVector2D",
			ValueFtype1< vector< vector< double > > >::global(),
			GFCAST( &Interpol2D::getTableVector2D ),
			RFCAST( &Interpol2D::setTableVector2D )
		),
	///////////////////////////////////////////////////////
	// Shared message definitions
	///////////////////////////////////////////////////////
		new SharedFinfo(
			"lookupReturn2D",
			lookupReturnShared, sizeof( lookupReturnShared ) / sizeof( Finfo * ),
			"This is a shared message for doing lookups on the table. "
			"Receives 2 doubles: x, y. "
			"Sends back a double with the looked-up z value." ),
		
	///////////////////////////////////////////////////////
	// MsgSrc definitions
	///////////////////////////////////////////////////////

	///////////////////////////////////////////////////////
	// MsgDest definitions
	///////////////////////////////////////////////////////
	};

	static string doc[] =
	{
		"Name", "Interpol2D",
		"Author", "Niraj Dudani, 2009, NCBS",
		"Description", "Interpol2D: Interpolation class. "
				"Handles lookup from a 2-dimensional grid of real-numbered values. "
				"Returns 'z' value based on given 'x' and 'y' values. "
				"Can either use interpolation or roundoff to the nearest index.",
	};
	
	static Cinfo interpol2DCinfo(
		doc,
		sizeof( doc ) / sizeof( string ),	
		initInterpolCinfo(),
		interpol2DFinfos,
		sizeof( interpol2DFinfos ) / sizeof( Finfo * ),
		ValueFtype1< Interpol2D >::global()
	);

	return &interpol2DCinfo;
}

static const Cinfo* interpol2DCinfo = initInterpol2DCinfo();

static const Slot lookupReturnSlot = 
	initInterpol2DCinfo()->getSlot( "lookupReturn2D.trig" );

////////////////////////////////////////////////////////////////////
// Here we set up Interpol2D value fields
////////////////////////////////////////////////////////////////////

void Interpol2D::setYmin( const Conn* c, double ymin ) 
{
	static_cast< Interpol2D* >( c->data() )->localSetYmin( ymin );
}

double Interpol2D::getYmin( Eref e )
{
	return static_cast< Interpol2D* >( e.data() )->ymin_;
}

void Interpol2D::setYmax( const Conn* c, double ymax ) 
{
	static_cast< Interpol2D* >( c->data() )->localSetYmax( ymax );
}

double Interpol2D::getYmax( Eref e )
{
	return static_cast< Interpol2D* >( e.data() )->ymax_;
}

void Interpol2D::setYdivs( const Conn* c, int ydivs ) 
{
	static_cast< Interpol2D* >( c->data() )->localSetYdivs( ydivs );
}

int Interpol2D::getYdivs( Eref e )
{
	return static_cast< Interpol2D* >( e.data() )->localGetYdivs( );
}

void Interpol2D::setDy( const Conn* c, double dy ) 
{
	static_cast< Interpol2D* >( c->data() )->localSetDy( dy );
}

double Interpol2D::getDy( Eref e )
{
	return static_cast< Interpol2D* >( e.data() )->localGetDy( );
}

void Interpol2D::setTable2D(
	const Conn* c,
	double val,
	const vector< unsigned int >& i )
{
	static_cast< Interpol2D* >( c->data() )->setTableValue( val, i );
}

double Interpol2D::getTable2D(
	Eref e,
	const vector< unsigned int >& i )
{
	return static_cast< Interpol2D* >( e.data() )->getTableValue( i );
}

void Interpol2D::setTableVector2D( const Conn* c, vector< vector< double > > value ) 
{
	static_cast< Interpol2D* >( c->data() )->localSetTableVector( value );
}

vector< vector< double > > Interpol2D::getTableVector2D( Eref e )
{
	return static_cast< Interpol2D* >( e.data() )->table_;
}

////////////////////////////////////////////////////////////////////
// Here we set up Interpol2D Destination functions
////////////////////////////////////////////////////////////////////

/**
 * lookupReturn uses its argument to do an interpolating lookup of the
 * table. It sends a return message to the
 * originating object with the looked up value.
 */
void Interpol2D::lookupReturn( const Conn* c, double v1, double v2 )
{
	double ret =
			static_cast< Interpol2D* >( c->data() )->innerLookup( v1, v2 );
	sendBack1< double >( c, lookupReturnSlot, ret );
}

////////////////////////////////////////////////////////////////////
// Here we set up private Interpol2D class functions.
////////////////////////////////////////////////////////////////////

Interpol2D::Interpol2D( 
	unsigned long xdivs, double xmin, double xmax,
	unsigned long ydivs, double ymin, double ymax )
		: Interpol( xdivs, xmin, xmax ),
		  ymin_( ymin ), ymax_( ymax )
{
	resize( xdivs + 1, ydivs + 1 );
	mode_ = 1; // Mode 1 is linear interpolation. 0 is indexing.
	
	if ( fabs( xmax_ - xmin_ ) > EPSILON )
		invDx_ = xdivs / ( xmax_ - xmin_);
	else
		invDx_ = 1.0;
	
	if ( fabs( ymax_ - ymin_ ) > EPSILON )
		invDy_ = ydivs / ( ymax_ - ymin_);
	else
		invDy_ = 1.0;
}

double Interpol2D::indexWithoutCheck( double x, double y ) const
{
	assert( table_.size() > 1 );
	
	unsigned long xInteger = static_cast< unsigned long >( ( x - xmin_ ) * invDx_ );
	assert( xInteger < table_.size() );
	
	unsigned long yInteger = static_cast< unsigned long >( ( y - ymin_ ) * invDy_ );
	assert( yInteger < table_[ 0 ].size() );
	
	return table_[ xInteger ][ yInteger ];
}

/**
 * Performs bi-linear interpolation, without bounds-checking.
 */
double Interpol2D::interpolateWithoutCheck( double x, double y ) const
{
	assert( table_.size() > 1 );
	
	double xv = ( x - xmin_ ) * invDx_;
	unsigned long xInteger = static_cast< unsigned long >( xv );
	double xFraction = xv - xInteger;
	assert( xInteger < table_.size() - 1 );
	
	double yv = ( y - ymin_ ) * invDy_;
	unsigned long yInteger = static_cast< unsigned long >( yv );
	double yFraction = yv - yInteger;
	assert( yInteger < table_[ 0 ].size() - 1 );
	
	/* The following is the same as:
			double z00 = table_[ xInteger ][ yInteger ];
			double z01 = table_[ xInteger ][ yInteger + 1 ];
			double z10 = table_[ xInteger + 1 ][ yInteger ];
			double z11 = table_[ xInteger + 1 ][ yInteger + 1 ];
	*/
	vector< vector< double > >::const_iterator iz0 = table_.begin() + xInteger;
	vector< double >::const_iterator iz00 = iz0->begin() + yInteger;
	vector< double >::const_iterator iz10 = ( iz0 + 1 )->begin() + yInteger;
	double z00 = *iz00;
	double z01 = *( iz00 + 1 );
	double z10 = *iz10;
	double z11 = *( iz10 + 1 );
	
	/* The following is the same as:
			return (
				z00 * ( 1 - xFraction ) * ( 1 - yFraction ) +
				z10 * xFraction * ( 1 - yFraction ) +
				z01 * ( 1 - xFraction ) * yFraction +
				z11 * xFraction * yFraction );
	*/
	double xFyF = xFraction * yFraction;
	return (
		z00 * ( 1 - xFraction - yFraction + xFyF ) +
		z10 * ( xFraction - xFyF ) +
		z01 * ( yFraction - xFyF ) +
		z11 * xFyF );
}

double Interpol2D::innerLookup( double x, double y ) const
{
	bool isOutOfBounds = false;
	
	if ( table_.size() == 0 )
		return 0.0;
	
	if ( x < xmin_ ) {
		x = xmin_;
		isOutOfBounds = true;
	}
	if ( x > xmax_ ) {
		x = xmax_;
		isOutOfBounds = true;
	}
	if ( y < ymin_ ) {
		y = ymin_;
		isOutOfBounds = true;
	}
	if ( y > ymax_ ) {
		y = ymax_;
		isOutOfBounds = true;
	}
	
	if ( mode_ == 0 || isOutOfBounds )
		return indexWithoutCheck( x, y );
	else 
		return interpolateWithoutCheck( x, y );
}

bool Interpol2D::operator==( const Interpol2D& other ) const
{
	return (
		xmin_ == other.xmin_ &&
		xmax_ == other.xmax_ &&
		ymin_ == other.ymin_ &&
		ymax_ == other.ymax_ &&
		mode_ == other.mode_ &&
		table_ == other.table_ );
}

bool Interpol2D::operator<( const Interpol2D& other ) const
{
	if ( table_.size() < other.table_.size() )
		return 1;
	
	if ( table_.size() > other.table_.size() )
		return 0;
	
	for ( size_t i = 0; i < table_.size(); i++ ) {
		for ( size_t j = 0; j < table_[ i ].size(); j++ ) {
			if ( table_[ i ][ j ] < other.table_[ i ][ j ] )
				return 1;
			if ( table_[ i ][ j ] > other.table_[ i ][ j ] )
				return 0;
		}
	}
	
	return 0;
}

void Interpol2D::localSetXdivs( int value ) {
	if ( value > 0 ) {
		this->resize( value + 1, ydivs() + 1 );
		invDx_ = value / ( xmax_ - xmin_ );
		return;
	}
	
	cerr << "Error: Interpol2D::localSetXdivs: # of divs should be >= 1.\n";
}

int Interpol2D::localGetXdivs( ) const {
	return xdivs();
}

void Interpol2D::localSetYmin( double value ) {
	if ( fabs( ymax_ - value) > EPSILON ) {
		ymin_ = value;
		invDy_ = ydivs() / ( ymax_ - ymin_ );
	} else {
		cerr << "Error: Interpol2D::localSetYmin: Ymin ~= Ymax : Assignment failed\n";
	}
}

void Interpol2D::localSetYmax( double value ) {
	if ( fabs( value - ymin_ ) > EPSILON ) {
		ymax_ = value;
		invDy_ = ydivs() / ( ymax_ - ymin_ );
	} else {
		cerr << "Error: Interpol2D::localSetYmax: Ymin ~= Ymax : Assignment failed\n";
	}
}

void Interpol2D::localSetYdivs( int value ) {
	if ( value > 0 ) {
		this->resize( xdivs() + 1, value + 1 );
		invDy_ = value / ( ymax_ - ymin_ );
		return;
	}
	
	cerr << "Error: Interpol2D::localSetYdivs: # of divs should be >= 1.\n";
}

int Interpol2D::localGetYdivs( ) const {
	return ydivs();
}

/**
 * \todo Later do interpolation etc to preserve contents.
 * \todo Later also check that it is OK for xmax_ < xmin_
 */
void Interpol2D::localSetDy( double value ) {
	if ( fabs( value ) - EPSILON > 0 ) {
		unsigned int ydivs = static_cast< unsigned int >( 
			0.5 + fabs( ymax_ - ymin_ ) / value );
		if ( ydivs < 1 || ydivs > MAX_DIVS ) {
			cerr <<
				"Error: Interpol2D::localSetDy Out of range:" <<
				ydivs + 1 << " entries in table.\n";
				return;
		}
		
		localSetYdivs( ydivs );
		invDy_ = ydivs / ( ymax_ - ymin_ );
	}
}

double Interpol2D::localGetDy() const {
	if ( ydivs() == 0 )
		return 0.0;
	else
		return ( ymax_ - ymin_ ) / ydivs();
}

void Interpol2D::localSetSy( double value ) {
	if ( fabs( value ) - EPSILON > 0 ) {
		double ratio = value / sy_;
		vector< vector< double > >::iterator i;
		vector< double >::iterator j;
		for ( i = table_.begin(); i != table_.end(); i++ )
			for ( j = i->begin(); j != i->end(); j++ )
				*j *= ratio;
		sy_ = value;
	} else {
		cerr << "Error: Interpol2D::localSetSy: sy too small:" <<
			value << "\n";
	}
}

void Interpol2D::setTableValue(
	double value,
	const vector< unsigned int >& index )
{
	assert( index.size() == 2 );
	unsigned int i0 = index[ 0 ];
	unsigned int i1 = index[ 1 ];
	
	if ( i0 < table_.size() && i1 < table_[ 0 ].size() )
		table_[ i0 ][ i1 ] = value;
	else
		cerr << "Error: Interpol2D::setTableValue: Index out of bounds!\n";
}

double Interpol2D::getTableValue(
	const vector< unsigned int >& index )
{
	assert( index.size() == 2 );
	unsigned int i0 = index[ 0 ];
	unsigned int i1 = index[ 1 ];
	
	if ( i0 < table_.size() && i1 < table_[ 0 ].size() )
		return table_[ i0 ][ i1 ];
	else {
		cerr << "Error: Interpol2D::getTableValue: Index out of bounds!\n";
		return 0.0;
	}
}

// This sets the whole thing up: values, xdivs, dx and so on. Only xmin
// and xmax are unknown to the input vector.
void Interpol2D::localSetTableVector( const vector< vector< double > >& value ) 
{
	int xsize = value.size();
	
	if ( xsize == 1 ) {
		cerr <<
			"Error: Interpol2D::localSetTableVector: Too few entries. "
			"Need at least 2x2 table. Not changing anything.\n";
		return;
	}
	
	if ( xsize == 0 ) {
		table_.resize( 0 );
		invDy_ = 1.0;
		invDx_ = 1.0;
		return;
	}
	
	unsigned int ysize = value[ 0 ].size();
	vector< vector< double > >::const_iterator i;
	for ( i = value.begin() + 1; i != value.end(); i++ )
		if ( i->size() != ysize ) {
			ysize = ~0u;
			break;
		}
	
	if ( ysize == ~0u ) {
		cerr <<
			"Error: Interpol2D::localSetTableVector: All rows should have a "
			"uniform width. Not changing anything.\n";
		return;
	}
	
	if ( ysize == 1 ) {
		cerr <<
			"Error: Interpol2D::localSetTableVector: Too few entries. "
			"Need at least 2x2 table. Not changing anything.\n";
		return;
	}
	
	if ( ysize == 0 ) {
		table_.resize( 0 );
		invDy_ = 1.0;
		invDx_ = 1.0;
		return;
	}
	
	table_ = value;
	invDx_ = xdivs() / ( xmax_ - xmin_ );
	invDy_ = ydivs() / ( ymax_ - ymin_ );
}

// This sets the whole thing up: values, xdivs, dx and so on. Only xmin
// and xmax are unknown to the input vector.
void Interpol2D::localAppendTableVector(
	const vector< vector< double > >& value ) 
{
	if ( value.empty() )
		return;
	
	unsigned int ysize = value[ 0 ].size();
	vector< vector< double > >::const_iterator i;
	for ( i = value.begin() + 1; i != value.end(); i++ )
		if ( i->size() != ysize ) {
			ysize = ~0u;
			break;
		}
	
	if ( ysize == ~0u ) {
		cerr <<
			"Error: Interpol2D::localAppendTableVector: All rows should have a "
			"uniform width. Not changing anything.\n";
		return;
	}
	
	if ( ! table_.empty() && ysize != table_[ 0 ].size() ) {
		cerr <<
			"Error: Interpol2D: localAppendTableVector: Table widths must match. "
			"Not changing anything.\n";
		return;
	}
	
	table_.insert( table_.end(), value.begin(), value.end() );
	invDx_ = xdivs() / ( xmax_ - xmin_ );
}

void Interpol2D::resize( unsigned int xsize, unsigned int ysize, double init ) {
	vector< vector< double > >::iterator i;
	table_.resize( xsize );
	for ( i = table_.begin(); i != table_.end(); i++ )
		i->resize( ysize, init );
	
	invDx_ = xdivs() / ( xmax_ - xmin_ );
	invDy_ = ydivs() / ( ymax_ - ymin_ );
}

void Interpol2D::innerPrint(
	const string& fname,
	bool appendFlag ) const
{
	std::ofstream fout;
	if ( appendFlag )
		fout.open( fname.c_str(), std::ios::app );
	else
		fout.open( fname.c_str(), std::ios::trunc );
	
	vector< vector< double > >::const_iterator i;
	vector< double >::const_iterator j;
	for ( i = table_.begin(); i != table_.end(); i++ ) {
		for ( j = i->begin(); j != i->end(); j++ )
			fout << *j << "\t";
		fout << "\n";
	}
	
	fout.close();
}

void Interpol2D::innerLoad( const string& fname, unsigned int skiplines )
{
	// Checking if xdivs/ydivs are different from default values. If they are,
	// then issue a warning.
	if ( xdivs() != 1 || ydivs() != 1 )
		cerr << "Warning: Interpol2D::innerLoad: Loading 2-D table from '" <<
			fname << "'. " <<
			"'xdivs' and 'ydivs' need not be specified. If you have set these fields, "
			"then they will be overridden while loading.\n";
	
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
		
		table_.clear( );
		unsigned int lastWidth = ~0u;
		double y;
		while( fin.good() ) {
			table_.resize( table_.size() + 1 );
			
			getline( fin, line );
			istringstream sstream( line );
			while( sstream >> y )
				table_.back().push_back( y );
			
			/*
			 * In case the last line of a file is blank.
			 */
			if ( table_.back().empty() ) {
				table_.pop_back();
				break;
			}
			
			if ( lastWidth != ~0u &&
			     table_.back().size() != lastWidth )
			{
				cerr << "Error: Interpol2D::innerLoad: " <<
					"In file " << fname <<
					", line " << table_.size() <<
					", row widths are not uniform! Will stop loading now.\n";
				table_.clear();
				return;
			}
			
			lastWidth = table_.back().size();
		}
		
		invDx_ = xdivs() / ( xmax_ - xmin_ );
		invDy_ = ydivs() / ( ymax_ - ymin_ );
	} else {
		cerr << "Error: Interpol2D::innerLoad: Failed to open file " << 
			fname << endl;
	}
}

#ifdef DO_UNIT_TESTS
void testInterpol2D()
{
/*
	static const unsigned int XDIVS = 100;
	cout << "\nDoing Interpol2D tests";

	Element* i1 = interpol2DCinfo->create( Id::scratchId(), "i1" );
	Element* i2 = interpol2DCinfo->create( Id::scratchId(), "i2" );
	unsigned int i;
	double ret = 0;

	set< int >( i1, "xdivs", XDIVS );

	set< double >( i1, "xmin", 0 );
	get< double >( i1, "xmin", ret );
	ASSERT( ret == 0.0, "testInterpol2D" );

	set< double >( i1, "xmax", 20 );
	get< double >( i1, "xmax", ret );
	ASSERT( ret == 20.0, "testInterpol2D" );

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
		Eref( i1 ).add( "lookupSrc", i2, "xmin" ), "connecting interpol2Ds" );

		//	i1->findFinfo( "lookupSrc" )->add( i1, i2, i2->findFinfo( "xmin" ) ), "connecting interpol2Ds"

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
*/
}

#endif // DO_UNIT_TESTS
