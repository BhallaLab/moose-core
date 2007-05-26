/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2006 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include <vector>
#include <iostream>

using namespace std;

#include "SparseMatrix.h"

const unsigned int SparseMatrix::MAX_ROWS = 200;
const unsigned int SparseMatrix::MAX_COLUMNS = 200;

ostream& operator <<( ostream& s, SparseMatrix& m )
{
	for ( unsigned int i = 0; i < m.nrows_; i++) {
		s << i << ":	";
		for ( unsigned int j = 0; j < m.ncolumns_; j++) {
			s.width( 4 );
			s << m.N_[ i * m.ncolumns_ + j ];
			if ( j == m.ncolumns_ - 1 )
				s << "\n";
		}
	}
	s << "\n";

	return s;
}

void SparseMatrix::setSize( unsigned int nrows, unsigned int ncolumns )
{
	if ( nrows < MAX_ROWS && ncolumns < MAX_COLUMNS ) {
		N_.resize( nrows * ncolumns );
		nrows_ = nrows;
		ncolumns_ = ncolumns;
	} else {
		cout << "Error: SparseMatrix::setSize( " << 
			nrows << ", " << ncolumns << ") out of range: ( " <<
			MAX_ROWS << ", " << MAX_COLUMNS << ")\n";
	}
}

void SparseMatrix::set( 
	unsigned int row, unsigned int column, int value )
{
	if ( row < nrows_ && column < ncolumns_ ) {
		int i = row * ncolumns_ + column;
		N_[ i ] = value;
	} else {
		cout << "Error: SparseMatrix::set( " << 
			row << ", " << column << ") is out of range: ( " <<
			nrows_ << ", " << ncolumns_ << ")\n";
	}
}

int SparseMatrix::get( unsigned int row, unsigned int column )
{
	if ( row < nrows_ && column < ncolumns_ ) {
		int i = row * ncolumns_ + column;
		return N_[ i ];
	} else {
		cout << "Error: SparseMatrix::set( " << 
			row << ", " << column << ") is out of range: ( " <<
			nrows_ << ", " << ncolumns_ << ")\n";
	}
	return 0;
}

double SparseMatrix::computeRowRate( 
	unsigned int row, const vector< double >& v
) const
{
	if ( row >= nrows_ ) {
		cout << "Error: SparseMatrix::computeRowRate: row out of bounds: " << 
			row << " >= " << nrows_ << "\n";
		return 0.0;
	}
	
	if ( v.size() != ncolumns_ ) {
		cout << "Error: SparseMatrix::computeRowRate: v.size() != ncolumns:" << 
			v.size() << " != " << ncolumns_ << "\n";
		return 0.0;
	}

	vector< int >::const_iterator i = N_.begin() + ncolumns_ * row;
	vector< double >::const_iterator j;
	double ret = 0.0;

	for ( j = v.begin(); j != v.end(); j++ )
		ret += *j * *i++;

	return ret;
}


#ifdef DO_UNIT_TESTS
#include "header.h"

void testSparseMatrix()
{
	cout << "\nTesting SparseMatrix" << flush;
	const unsigned int NR = 4;
	const unsigned int NC = 5;

	SparseMatrix sm( NR, NC);

	for ( unsigned int i = 0; i < NR; i++ ) {
		for ( unsigned int j = 0; j < NC; j++ ) {
			sm.set( i, j, 10 * i + j );
			unsigned int ret = sm.get( i, j );
			ASSERT( ret == 10 * i + j, "set/get" );
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
