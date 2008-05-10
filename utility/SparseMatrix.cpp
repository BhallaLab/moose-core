/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2006 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

const unsigned int SM_MAX_ROWS = 10000;
const unsigned int SM_MAX_COLUMNS = 10000;
const unsigned int SM_RESERVE = 8;

#ifdef DO_UNIT_TESTS
#include <functional>
#include "../basecode/header.h"
#include "SparseMatrix.h"


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
	// cout << sm;
	//
	// vector< int >::const_iterator entry;
	// vector< unsigned int >::const_iterator colIndex;

	const int* entry;
	const unsigned int* colIndex;

	for ( unsigned int i = 0; i < NR; i++ ) {
		sm.getRow( i, &entry, &colIndex );
		for ( unsigned int j = 0; j < NC; j++ ) {
			if ( j != *colIndex++ )
				continue;
			int ret = 10 * i + j;
			ASSERT( ret == *entry++ , "getRow" );
		}
	}

	/*
	for ( unsigned int i = 0; i < NC; i++ ) {
		sm.getColumn( i, entry, colIndex );
		for ( unsigned int j = 0; j < NR; j++ ) {
			if ( j != *colIndex++ )
				continue;
			int ret = 10 * i + j;
			ASSERT( ret == *entry++ , "getRow" );
		}
	}
	*/


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
