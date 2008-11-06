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
const unsigned int KinSparseMatrix::MAX_ROWS = 100000;
const unsigned int KinSparseMatrix::MAX_COLUMNS = 100000;

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
		return N_[ rowStart_[row] + (i - begin) ];
	}
}

/**
 * Returns all non-zero column indices, for the specified row.  
 * This gives reac #s in orig matrix, and molecule #s in the 
 * transposed matrix
 */
int KinSparseMatrix::getRowIndices( unsigned int row, 
	vector< unsigned int >& indices )
{
	indices.resize( 0 );
	indices.insert( indices.end(), 
		colIndex_.begin() + rowStart_[ row ],
		colIndex_.begin() + rowStart_[ row + 1 ] );
	return ( indices.size() );
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

/**
 * Puts a transpose of current matrix into ret.
 */
void KinSparseMatrix::transpose( KinSparseMatrix& ret ) const
{
	ret.N_.resize( 0 );
	ret.colIndex_.resize( 0 );
	ret.rowStart_.resize( 0 );
	ret.nrows_ = ncolumns_;
	ret.ncolumns_ = nrows_;
	// vector< unsigned int > currentColumn( nrows_ + 1, 0 );
	vector< unsigned int > currentRowStart( rowStart_ );

	ret.rowStart_.push_back( 0 );
	for ( unsigned int col = 0; col < ncolumns_; ++col ) {
		for ( unsigned int i = 0; i < nrows_; ++i ) {
			unsigned int j = currentRowStart[ i ];
			if ( j >= rowStart_[ i + 1 ] )
				continue; // This row has been completed.
			if ( colIndex_[ j ] == col ) {
				ret.N_.push_back( N_[ j ] );
				ret.colIndex_.push_back( i );
				++currentRowStart[ i ];
			} else {
				// nothing to do here.
			}
		}
		ret.rowStart_.push_back( ret.N_.size() );
	}
}

/**
 * Has to operate on transposed matrix
 * row argument refers to reac# in this transformed situation.
 * Fills up 'deps' with reac#s that depend on the row argument.
 * Does NOT ensure that list is unique.
 */
void KinSparseMatrix::getGillespieDependence( 
	unsigned int row, vector< unsigned int >& deps
) const
{
	deps.resize( 0 );
	// vector< unsigned int > deps;
	for ( unsigned int i = 0; i < nrows_; ++i ) {
		// i is index for reac # here. Note that matrix is transposed.
		unsigned int j = rowStart_[ row ];
		unsigned int jend = rowStart_[ row + 1 ];
		unsigned int k = rowStart_[ i ];
		unsigned int kend = rowStart_[ i + 1 ];
		
		while ( j < jend && k < kend ) {
			if ( colIndex_[ j ] == colIndex_[ k ] ) {
				if ( N_[ k ] < 0 ) {
					deps.push_back( i );
				}
				++j;
				++k;
			} else if ( colIndex_[ j ] < colIndex_[ k ] ) {
				++j;
			} else if ( colIndex_[ j ] > colIndex_[ k ] ) {
				++k;
			} else {
				assert( 0 );
			}
		}
	}
}

/**
 * This too operates on the transposed matrix, because we need to get all
 * the molecules for a given reac: a column in the original N matrix.
 */
void KinSparseMatrix::fireReac( unsigned int reacIndex, vector< double >& S ) 
	const
{
	assert( ncolumns_ == S.size() && reacIndex < nrows_ );
	unsigned int rowBeginIndex = rowStart_[ reacIndex ];
	// vector< int >::const_iterator rowEnd = N_.begin() + rowStart_[ reacIndex + 1];
	vector< int >::const_iterator rowBegin = 
		N_.begin() + rowBeginIndex;
	vector< int >::const_iterator rowEnd = 
		N_.begin() + rowTruncated_[ reacIndex ];
	vector< unsigned int >::const_iterator molIndex = 
		colIndex_.begin() + rowBeginIndex;

	for ( vector< int >::const_iterator i = rowBegin; i != rowEnd; ++i )
		S[ *molIndex++ ] += *i;
}

/**
 * This function generates a new internal list of rowEnds, such that
 * they are all less than the maxColumnIndex.
 * It is used because in fireReac we don't want to update all the 
 * molecules, only those that are variable.
 */
void KinSparseMatrix::truncateRow( unsigned int maxColumnIndex )
{
	rowTruncated_.resize( nrows_, 0 );
	if ( colIndex_.size() == 0 )
		return;
	for ( unsigned int i = 0; i < nrows_; ++i ) {
		unsigned int endCol = rowStart_[ i ];
		for ( unsigned int j = rowStart_[ i ]; 
			j < rowStart_[ i + 1 ]; ++j ) {
			if ( colIndex_[ j ] < maxColumnIndex ) {
				endCol = j + 1;
			} else {
				break;
			}
		}
		rowTruncated_[ i ] = endCol;
	}
}


void makeVecUnique( vector< unsigned int >& v )
{
	vector< unsigned int >::iterator pos = unique( v.begin(), v.end() );
	v.resize( pos - v.begin() );
}

#ifdef DO_UNIT_TESTS
#include "header.h"

void testKinSparseMatrix()
{
	// This is the stoichiometry matrix for the unidirectional reacns
	// coming out of the following system:
	// a + b <===> c
	// c + d <===> e
	// a + f <===> g
	// a + e <===> 2g
	//
	// When halfreac 0 fires, it affects 0, 1, 2, 4, 6.
	// and so on.
	static int transposon[][ 8 ] = { 
		{ -1,  1,  0,  0, -1,  1, -1,  1 },
		{ -1,  1,  0,  0,  0,  0,  0,  0 },
		{  1, -1, -1,  1,  0,  0,  0,  0 },
		{  0,  0, -1,  1,  0,  0,  0,  0 },
		{  0,  0,  1, -1,  0,  0, -1,  1 },
		{  0,  0,  0,  0, -1,  1,  0,  0 },
		{  0,  0,  0,  0,  1, -1,  2, -2 }
	};

	cout << "\nTesting KinSparseMatrix" << flush;
	const unsigned int NR = 4;
	const unsigned int NC = 5;

	const unsigned int NTR = 7; // for transposon
	const unsigned int NTC = 8;

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

	////////////////////////////////////////////////////////////////
	// Checking transposition operation
	////////////////////////////////////////////////////////////////
	KinSparseMatrix orig( NTR, NTC );
	for ( unsigned int i = 0; i < NTR; i++ )
		for ( unsigned int j = 0; j < NTC; j++ )
			orig.set( i, j, transposon[ i ][ j ] );

	ASSERT( orig.rowStart_.size() == 8, "transposed: rowStart" );
	ASSERT( orig.rowStart_[0] == 0, "transposon: rowStart" );
	ASSERT( orig.rowStart_[1] == 6, "transposon: rowStart" );
	ASSERT( orig.rowStart_[2] == 8, "transposon: rowStart" );
	ASSERT( orig.rowStart_[3] == 12, "transposon: rowStart" );
	ASSERT( orig.rowStart_[4] == 14, "transposon: rowStart" );
	ASSERT( orig.rowStart_[5] == 18, "transposon: rowStart" );
	ASSERT( orig.rowStart_[6] == 20, "transposon: rowStart" );
	ASSERT( orig.rowStart_[7] == 24, "transposon: rowStart" );

	ASSERT( orig.colIndex_[0] == 0, "transposon: colIndex" );
	ASSERT( orig.colIndex_[1] == 1, "transposon: colIndex" );
	ASSERT( orig.colIndex_[2] == 4, "transposon: colIndex" );
	ASSERT( orig.colIndex_[3] == 5, "transposon: colIndex" );
	ASSERT( orig.colIndex_[4] == 6, "transposon: colIndex" );
	ASSERT( orig.colIndex_[5] == 7, "transposon: colIndex" );
	ASSERT( orig.colIndex_[6] == 0, "transposon: colIndex" );
	ASSERT( orig.colIndex_[7] == 1, "transposon: colIndex" );

	ASSERT( orig.N_[0] == -1, "transposon: N" );
	ASSERT( orig.N_[1] == 1, "transposon: N" );
	ASSERT( orig.N_[2] == -1, "transposon: N" );
	ASSERT( orig.N_[3] == 1, "transposon: N" );
	ASSERT( orig.N_[4] == -1, "transposon: N" );
	ASSERT( orig.N_[5] == 1, "transposon: N" );
	ASSERT( orig.N_[6] == -1, "transposon: N" );
	ASSERT( orig.N_[7] == 1, "transposon: N" );

	KinSparseMatrix trans( NTC, NTR );
	orig.transpose( trans );

	ASSERT( trans.rowStart_.size() == 9, "transposed: rowStart" );
	ASSERT( trans.rowStart_[0] == 0, "transposed: rowStart" );
	ASSERT( trans.rowStart_[1] == 3, "transposed: rowStart" );
	ASSERT( trans.rowStart_[2] == 6, "transposed: rowStart" );
	ASSERT( trans.rowStart_[3] == 9, "transposed: rowStart" );
	ASSERT( trans.rowStart_[4] == 12, "transposed: rowStart" );
	ASSERT( trans.rowStart_[5] == 15, "transposed: rowStart" );
	ASSERT( trans.rowStart_[6] == 18, "transposed: rowStart" );
	ASSERT( trans.rowStart_[7] == 21, "transposed: rowStart" );
	ASSERT( trans.rowStart_[8] == 24, "transposed: rowStart" );

	ASSERT( trans.colIndex_[0] == 0, "transposed: colIndex" );
	ASSERT( trans.colIndex_[1] == 1, "transposed: colIndex" );
	ASSERT( trans.colIndex_[2] == 2, "transposed: colIndex" );
	ASSERT( trans.colIndex_[3] == 0, "transposed: colIndex" );
	ASSERT( trans.colIndex_[4] == 1, "transposed: colIndex" );
	ASSERT( trans.colIndex_[5] == 2, "transposed: colIndex" );
	ASSERT( trans.colIndex_[6] == 2, "transposed: colIndex" );
	ASSERT( trans.colIndex_[7] == 3, "transposed: colIndex" );

	ASSERT( trans.N_[0] == -1, "transposed: N" );
	ASSERT( trans.N_[1] == -1, "transposed: N" );
	ASSERT( trans.N_[2] == 1, "transposed: N" );
	ASSERT( trans.N_[3] == 1, "transposed: N" );
	ASSERT( trans.N_[4] == 1, "transposed: N" );
	ASSERT( trans.N_[5] == -1, "transposed: N" );
	ASSERT( trans.N_[6] == -1, "transposed: N" );
	ASSERT( trans.N_[7] == -1, "transposed: N" );

	for ( unsigned int i = 0; i < NTR; i++ ) {
		for ( unsigned int j = 0; j < NTC; j++ ) {
			int ret = trans.get( j, i );
			if ( transposon[ i ][ j ] != ret )
				ASSERT( 0, "transposed: N" );
		}
	}

	////////////////////////////////////////////////////////////////
	// Checking generation of dependency graphs.
	////////////////////////////////////////////////////////////////
	
	vector< unsigned int > deps;
	trans.getGillespieDependence( 0, deps );
	makeVecUnique( deps );
	ASSERT( deps.size() == 5, "Gillespie dependence" );
	ASSERT( deps[0] == 0, "Gillespie dependence" );
	ASSERT( deps[1] == 1, "Gillespie dependence" );
	ASSERT( deps[2] == 2, "Gillespie dependence" );
	ASSERT( deps[3] == 4, "Gillespie dependence" );
	ASSERT( deps[4] == 6, "Gillespie dependence" );

	trans.getGillespieDependence( 1, deps );
	makeVecUnique( deps );
	ASSERT( deps.size() == 5, "Gillespie dependence" );
	ASSERT( deps[0] == 0, "Gillespie dependence" );
	ASSERT( deps[1] == 1, "Gillespie dependence" );
	ASSERT( deps[2] == 2, "Gillespie dependence" );
	ASSERT( deps[3] == 4, "Gillespie dependence" );
	ASSERT( deps[4] == 6, "Gillespie dependence" );

	trans.getGillespieDependence( 2, deps );
	makeVecUnique( deps );
	ASSERT( deps.size() == 4, "Gillespie dependence" );
	ASSERT( deps[0] == 1, "Gillespie dependence" );
	ASSERT( deps[1] == 2, "Gillespie dependence" );
	ASSERT( deps[2] == 3, "Gillespie dependence" );
	ASSERT( deps[3] == 6, "Gillespie dependence" );

	trans.getGillespieDependence( 4, deps );
	makeVecUnique( deps );
	ASSERT( deps.size() == 5, "Gillespie dependence" );
	ASSERT( deps[0] == 0, "Gillespie dependence" );
	ASSERT( deps[1] == 4, "Gillespie dependence" );
	ASSERT( deps[2] == 5, "Gillespie dependence" );
	ASSERT( deps[3] == 6, "Gillespie dependence" );
	ASSERT( deps[4] == 7, "Gillespie dependence" );

	trans.getGillespieDependence( 6, deps );
	makeVecUnique( deps );
	ASSERT( deps.size() == 6, "Gillespie dependence" );
	ASSERT( deps[0] == 0, "Gillespie dependence" );
	ASSERT( deps[1] == 3, "Gillespie dependence" );
	ASSERT( deps[2] == 4, "Gillespie dependence" );
	ASSERT( deps[3] == 5, "Gillespie dependence" );
	ASSERT( deps[4] == 6, "Gillespie dependence" );
	ASSERT( deps[5] == 7, "Gillespie dependence" );
}

#endif
