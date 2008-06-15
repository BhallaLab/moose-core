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

RateLookup::RateLookup( double* base, RateLookupGroup* group )
{
	base_ = base;
	group_ = group;
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
	nPts_ = 1 + nDivs;
	dx_ = ( max - min ) / nDivs;
	nColumns_ = 2 * nSpecies; // Every row has 2 entries for each type of gate
	
	table_.resize( nPts_ * nColumns_ );
}

void RateLookupGroup::addTable(
	int species,
	const vector< double >& C1,
	const vector< double >& C2 )
{
	vector< double >::const_iterator ic1 = C1.begin();
	vector< double >::const_iterator ic2 = C2.begin();
	for ( unsigned int igrid = 0; igrid < nPts_ ; ++igrid ) {
		table_[ nColumns_ * igrid + 2 * species ] = *ic1;
		table_[ nColumns_ * igrid + 2 * species + 1 ] = *ic2;
		++ic1, ++ic2;
	}
}

RateLookup RateLookupGroup::slice( unsigned int species )
{
	return RateLookup( &table_[ 2 * species ], this );
}

void RateLookupGroup::getKey( double x, LookupKey& key )
{
	double div   = ( x - min_ ) / dx_;
	unsigned int integer  = ( unsigned int )( div );
	
	key.fraction = div - integer;
	key.offset1  = integer * nColumns_;
	key.offset2  = key.offset1 + nColumns_;
}
