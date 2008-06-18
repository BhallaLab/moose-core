/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2006 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "../basecode/header.h"
#include "SparseMatrix.h"
#include <functional>
const unsigned int SM_MAX_ROWS = 10000;
const unsigned int SM_MAX_COLUMNS = 10000;
const unsigned int SM_RESERVE = 8;

#ifdef DO_UNIT_TESTS



void testSparseMatrix()
{
	cout << "\nTesting SparseMatrix" << flush;
	const unsigned int NR = 4;
	const unsigned int NC = 5;

	SparseMatrix< int > sm( NR, NC);

	for ( unsigned int i = 0; i < NR; i++ ) {
		for ( unsigned int j = 0; j < NC; j++ ) {
			sm.set( i, j, 10 * i + j );
			// cout << i << ", " << j << ", " << sm.nColumns() << endl;
			int ret = sm.get( i, j );
			ASSERT( ret == static_cast< int >( 10 * i + j ), "set/get" );
		}
	}

	// Initially all entries are filled. Check this.
	ASSERT( sm.nEntries() == NR * NC, "Check initial fill" );

	// Now zap the 0,0 entry.
	sm.unset( 0 , 0 );
	ASSERT( sm.nEntries() == NR * NC - 1, "Check unset" );

	// cout << sm;
	//
	// vector< int >::const_iterator entry;
	// vector< unsigned int >::const_iterator colIndex;

	const int* entry;
	const unsigned int* colIndex;
	unsigned int numEntries = 0;
	vector< int > e;
	vector< unsigned int > ri;

	for ( unsigned int i = 0; i < NR; i++ ) {
		numEntries = sm.getRow( i, &entry, &colIndex );
		for ( unsigned int j = 0; j < NC; j++ ) {
			if ( j != *colIndex++ )
				continue;
			int ret = 10 * i + j;
			ASSERT( ret == *entry++ , "getRow" );
		}
	}

	for ( unsigned int i = 0; i < NC; i++ ) {
		unsigned int rowIndex = 0;

		numEntries = sm.getColumn( i, e, ri );
		for ( unsigned int j = 0; j < numEntries; j++ ) {
			rowIndex = ri[j];
			int ret = 10 * rowIndex + i;
			ASSERT( ret == e[j] , "getColumn" );
		}
	}

	sm.unset( 2, 0 );
	sm.unset( 2, 1 );
	sm.unset( 2, 2 );
	sm.unset( 2, 3 );
	sm.unset( 2, 4 );

	numEntries = sm.getRow( 1, &entry, &colIndex );
	ASSERT( numEntries == 5, "check Row" );
	ASSERT( entry[2] == 12, "check Row" );

	numEntries = sm.getRow( 2, &entry, &colIndex );
	ASSERT( numEntries == 0, "check Row" );

	numEntries = sm.getRow( 3, &entry, &colIndex );
	ASSERT( numEntries == 5, "check Row" );
	ASSERT( entry[3] == 33, "check Row" );

	numEntries = sm.getColumn( 0, e, ri );
	ASSERT( numEntries == 2, "check Column" );
	ASSERT( e[0] == 10, "check Column" );
	ASSERT( e[1] == 30, "check Column" );

	numEntries = sm.getColumn( 2, e, ri );
	ASSERT( numEntries == 3, "check Column" );
	ASSERT( e[0] == 2, "check Column" );
	ASSERT( e[1] == 12, "check Column" );
	ASSERT( e[2] == 32, "check Column" );

	/*
	vector< double > v( 5, 1.0 );
	double dret = sm.computeRowRate( 0, v );
	ASSERT( dret == 10.0, "computeRowRate" );
	dret = sm.computeRowRate( 1, v );
	ASSERT( dret == 60.0, "computeRowRate" );
	dret = sm.computeRowRate( 2, v );
	ASSERT( dret == 110.0, "computeRowRate" );
	dret = sm.computeRowRate( 3, v );
	ASSERT( dret == 160.0, "computeRowRate" );
	*/
}

#endif
