/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**   copyright (C) 2003-2007 Upinder S. Bhalla, Niraj Dudani and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include <vector>
using namespace std;

#include "RateLookup.h"

/*
 * RateLookup function definitions
 */

RateLookup::RateLookup( double* base, RateLookupGroup* group, bool interpolate )
{
	base_ = base;
	group_ = group;
	interpolate_ = interpolate;
}

void RateLookup::getKey( double x, LookupKey& key )
{
	group_->getKey( x, key );
}

void RateLookup::rates( const LookupKey& key, double& C1, double& C2 )
{
	double a, b;
	double *ap, *bp;
	
	ap = base_ + key.offset1;
	
	if ( ! interpolate_ ) {
		C1 = *ap;
		C2 = *( ap + 1 );
		
		return;
	}
	
	bp = base_ + key.offset2;
	
	a = *ap;
	b = *bp;
	C1 = a + ( b - a ) * key.fraction;
	
	a = *( ap + 1 );
	b = *( bp + 1 );
	C2 = a + ( b - a ) * key.fraction;
}

/*
 * RateLookupGroup function definitions
 */

RateLookupGroup::RateLookupGroup(
	double min, double max, unsigned int nDivs, unsigned int nSpecies )
{
	min_ = min;
	max_ = max;
	// Number of points is 1 more than number of divisions.
	// Then add one more since we may interpolate at the last point in the table.
	nPts_ = nDivs + 1 + 1;
	dx_ = ( max - min ) / nDivs;
	// Every row has 2 entries for each type of gate
	nColumns_ = 2 * nSpecies;
	
	interpolate_.resize( nSpecies );
	table_.resize( nPts_ * nColumns_ );
}

void RateLookupGroup::addTable(
	int species,
	const vector< double >& C1,
	const vector< double >& C2,
	bool interpolate )
{
	vector< double >::const_iterator ic1 = C1.begin();
	vector< double >::const_iterator ic2 = C2.begin();
	vector< double >::iterator iTable = table_.begin() + 2 * species;
	// Loop until last but one point
	for ( unsigned int igrid = 0; igrid < nPts_ - 1 ; ++igrid ) {
		*( iTable )     = *ic1;
		*( iTable + 1 ) = *ic2;
		
		iTable += nColumns_;
		++ic1, ++ic2;
	}
	// Then duplicate the last point
	*( iTable )     = C1.back();
	*( iTable + 1 ) = C2.back();
	
	interpolate_[ species ] = interpolate;
}

RateLookup RateLookupGroup::slice( unsigned int species )
{
	return RateLookup( &table_[ 2 * species ], this, interpolate_[ species ] );
}

void RateLookupGroup::getKey( double x, LookupKey& key )
{
	if ( x < min_ )
		x = min_;
	else if ( x > max_ )
		x = max_;
	
	double div = ( x - min_ ) / dx_;
	unsigned int integer = ( unsigned int )( div );
	
	key.fraction = div - integer;
	key.offset1  = integer * nColumns_;
	key.offset2  = key.offset1 + nColumns_;
}
