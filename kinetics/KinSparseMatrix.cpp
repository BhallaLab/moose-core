/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2006 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include <algorithm>
#include <vector>
#include <iostream>
#include <cassert>
#include <math.h> // used for isnan
#include "../utility/utility.h" // isnan is undefined in VC++ and BC5, utility.h contains a workaround macro
using namespace std;

#include "KinSparseMatrix.h"

// Substantially bigger than possible using a full matrix.
const unsigned int KinSparseMatrix::MAX_ROWS = 10000;
const unsigned int KinSparseMatrix::MAX_COLUMNS = 10000;

KinSparseMatrix::KinSparseMatrix()
	: nrows_( 0 ), ncolumns_( 0 )
{
	N_.resize( 0 );
	N_.reserve( 16 );
	colIndex_.resize( 0 );
	colIndex_.reserve( 16 );
}


ostream& operator <<( ostream& s, KinSparseMatrix& m )
{
	for ( unsigned int i = 0; i < m.nrows_; i++) {
		unsigned int start = m.rowStart_[i];
		unsigned int end = m.rowStart_[i + 1];
		s << i << ":	";

		unsigned int k = start;
		int value = 0;
		for ( unsigned int j = 0; j < m.ncolumns_; j++ ) {
			if ( k == end ) 
				value = 0;
			else if ( j == m.colIndex_[ k ] )
				value = m.N_[ k++ ];
			else
				value = 0;

			s.width( 4 );
			s << value ;
		}
		s << "\n";
	}
	s << "\n";

	return s;
}

void KinSparseMatrix::setSize( unsigned int nrows, unsigned int ncolumns )
{
	if ( nrows < MAX_ROWS && ncolumns < MAX_COLUMNS ) {
		N_.resize( 0 );
		N_.reserve( 2 * nrows );
		nrows_ = nrows;
		ncolumns_ = ncolumns;
		rowStart_.resize( nrows + 1, 0 );
		colIndex_.resize( 0 );
		colIndex_.reserve( 2 * nrows );
	} else {
		cout << "Error: KinSparseMatrix::setSize( " << 
			nrows << ", " << ncolumns << ") out of range: ( " <<
			MAX_ROWS << ", " << MAX_COLUMNS << ")\n";
	}
}

/**
 * This function assumes that we rarely set any entry to zero: most
 * cases are when we add a new non-zero entry. If we were to use it to
 * exhaustively fill up all coords in the matrix it would be quite slow.
 */
void KinSparseMatrix::set( 
	unsigned int row, unsigned int column, int value )
{
	vector< unsigned int >::iterator i;
	vector< unsigned int >::iterator begin = 
		colIndex_.begin() + rowStart_[ row ];
	vector< unsigned int >::iterator end = 
		colIndex_.begin() + rowStart_[ row + 1 ];

	if ( begin == end ) { // Entire row was empty.
		if ( value == 0 ) // Don't need to change an already zero entry
			return;
		unsigned long offset = begin - colIndex_.begin();
		colIndex_.insert( colIndex_.begin() + offset, column );
		N_.insert( N_.begin() + offset, value );
		for ( unsigned int j = row + 1; j <= nrows_; j++ )
			rowStart_[ j ]++;
		return;
	}

	if ( column > *( end - 1 ) ) { // add entry at end of row.
		if ( value == 0 )
			return;
		unsigned long offset = end - colIndex_.begin();
		colIndex_.insert( colIndex_.begin() + offset, column );
		N_.insert( N_.begin() + offset, value );
		for ( unsigned int j = row + 1; j <= nrows_; j++ )
			rowStart_[ j ]++;
		return;
	}
	for ( i = begin; i != end; i++ ) {
		if ( *i == column ) { // Found desired entry. By defn it is nonzero.
			if ( value != 0 ) // Assign value
				N_[ i - colIndex_.begin()] = value;
			else { // Clear out value and entry.
				unsigned long offset = i - colIndex_.begin();
				colIndex_.erase( i );
				N_.erase( N_.begin() + offset );
				for ( unsigned int j = row + 1; j <= nrows_; j++ )
					rowStart_[ j ]--;
			}
			return;
		} else if ( *i > column ) { // Desired entry is blank.
			if ( value == 0 ) // Don't need to change an already zero entry
				return;
			unsigned long offset = i - colIndex_.begin();
			colIndex_.insert( colIndex_.begin() + offset, column );
			N_.insert( N_.begin() + offset, value );
			for ( unsigned int j = row + 1; j <= nrows_; j++ )
				rowStart_[ j ]++;
			return;
		}
	}
}

int KinSparseMatrix::get( unsigned int row, unsigned int column )
{
	assert( row < nrows_ && column < ncolumns_ );
	vector< unsigned int >::iterator i;
	vector< unsigned int >::iterator begin = 
		colIndex_.begin() + rowStart_[ row ];
	vector< unsigned int >::iterator end = 
		colIndex_.begin() + rowStart_[ row + 1 ];

	i = find( begin, end, column );
	if ( i == end ) { // most common situation for a sparse Stoich matrix.
		return 0;
	} else {
		return N_[ rowStart_[row] + i - begin ];
	}
}

double KinSparseMatrix::computeRowRate( 
	unsigned int row, const vector< double >& v
) const
{
	assert( row < nrows_ );
	assert( v.size() == ncolumns_ );

	vector< int >::const_iterator i;
	unsigned int rs = rowStart_[ row ];
	vector< unsigned int >::const_iterator j = colIndex_.begin() + rs;
	vector< int >::const_iterator end = N_.begin() + rowStart_[ row + 1 ];
	
	double ret = 0.0;
	for ( i = N_.begin() + rs; i != end; i++ )
		ret += *i * v[ *j++ ];

	// assert ( !( ret !<>= 0.0 ) );
	assert ( !( isnan( ret ) ) );
	return ret;
}


#ifdef DO_UNIT_TESTS
#include "header.h"

void testKinSparseMatrix()
{
	cout << "\nTesting KinSparseMatrix" << flush;
	const unsigned int NR = 4;
	const unsigned int NC = 5;

	KinSparseMatrix sm( NR, NC);

	for ( unsigned int i = 0; i < NR; i++ ) {
		for ( unsigned int j = 0; j < NC; j++ ) {
			sm.set( i, j, 10 * i + j );
			// cout << i << ", " << j << ", " << sm.nColumns() << endl;
			int ret = sm.get( i, j );
			ASSERT( ret == static_cast< int >( 10 * i + j ), "set/get" );
		}
	}
	// cout << sm;

	vector< double > v( 5, 1.0 );
	double dret = sm.computeRowRate( 0, v );
	ASSERT( dret == 10.0, "computeRowRate" );
	dret = sm.computeRowRate( 1, v );
	ASSERT( dret == 60.0, "computeRowRate" );
	dret = sm.computeRowRate( 2, v );
	ASSERT( dret == 110.0, "computeRowRate" );
	dret = sm.computeRowRate( 3, v );
	ASSERT( dret == 160.0, "computeRowRate" );
}

#endif
