/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2005 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
********************************************************************* */

#include <vector>
#include <iostream>
#include <math.h>
using namespace std;

#include "Interpol.h"

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
	// unsigned long i = static_cast< unsigned long >( xv );
	// return table_[ i ] + ( table_[ i + 1 ] - table_ [ i ] ) * ( xv - i );
	vector< double >::const_iterator i = table_.begin() + 
		static_cast< unsigned long >( xv );
	return *i + ( *( i + 1 ) - *i ) * ( xv - floor( xv ) ); 
}

double Interpol::doLookup( double x ) const
{
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
		cerr << "Warning: InterpolWrapper: Xmin ~= Xmax : Assignment failed\n";
	}
}
void Interpol::localSetXmax( double value ) {
	if ( fabs( value - xmin_ ) > EPSILON ) {
		xmax_ = value;
		invDx_ = static_cast< double >( table_.size() - 1 ) / 
			( xmax_ - xmin_ );
	} else {
		cerr << "Warning: InterpolWrapper: Xmin ~= Xmax : Assignment failed\n";
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
		int xdivs = static_cast< int >( 
			0.5 + fabs( xmax_ - xmin_ ) / value );
		if ( xdivs < 1 || xdivs > MAX_DIVS ) {
			cerr << "Warning: InterpolWrapper: Out of range:" <<
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
		cerr << "Warning: InterpolWrapper: localSetSy: sy too small:" <<
			value << "\n";
	}
}

void Interpol::setTableValue( double value, int index ) {
	if ( index >= 0 && 
		static_cast< unsigned int >( index ) < table_.size() )
		table_[ index ] = value;
}

double Interpol::getTableValue( int index ) const {
	if ( index >= 0 && 
		static_cast< unsigned int >( index ) < table_.size() )
		return table_[ index ];
	return 0.0;
}

void Interpol::tabFill( int xdivs, int mode )
{
	vector< double > newtab;
	newtab.resize( xdivs + 1, 0.0 );
	double dx = ( xmax_ - xmin_ ) / static_cast< double >( xdivs );
	mode_ = 1; // Has to be, so we can interpolate.
	for ( int i = 0; i <= xdivs; i++ )
		newtab[ i ] = doLookup(
			xmin_ + dx * static_cast< double >( i ) );
	table_ = newtab;
	mode_ = mode;
	invDx_ = 1.0/dx;
}
