/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2014 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include <vector>
#include <algorithm>
#include <cassert>
#include <functional>
#include <iostream>
#include <iomanip>
#include <gsl/gsl_linalg.h>
//#include "/usr/include/gsl/gsl_linalg.h"
using namespace std;
#include "../basecode/SparseMatrix.h"
#include "FastMatrixElim.h"

/*
const unsigned int SM_MAX_ROWS = 200000;
const unsigned int SM_MAX_COLUMNS = 200000;
const unsigned int SM_RESERVE = 8;
*/

void sortByColumn( 
			vector< unsigned int >& col, vector< double >& entry );
void testSorting();

// 	
//	static unsigned int parents[] = { 1,6,3,6,5,8,7,8,9,10,-1};
//	unsigned int numKids[] = {0,1,0,1,0,2,

/**
 * Reorders rows and columns to put the matrix in the form suitable for 
 * rapid single-pass inversion. Returns 0 on failure, but at this
 * point I don't have a proper test for this.
 */

static const unsigned int EMPTY_VOXEL(-1);
bool FastMatrixElim::hinesReorder( const vector< unsigned int >& parentVoxel )
{
	// First we fill in the vector that specifies the old row number 
	// assigned to each row of the reordered matrix.
	assert( parentVoxel.size() == nrows_ );
	vector< unsigned int > numKids( nrows_, 0 );
	vector< unsigned int > lookupOldRowFromNew;
	vector< bool > rowPending( nrows_, true );
	unsigned int numDone = 0;
	for ( unsigned int i = 0; i < nrows_; ++i ) {
		if ( parentVoxel[i] != EMPTY_VOXEL )
			numKids[ parentVoxel[i] ]++;
	}
	while ( numDone < nrows_ ) {
		for ( unsigned int i = 0; i < nrows_; ++i ) {
			if ( rowPending[i] && numKids[i] == 0 ) {
				lookupOldRowFromNew.push_back( i );
				rowPending[i] = false;
				numDone++;
				unsigned int pa = parentVoxel[i];
				// Unsure what the root parent is. Assume it is -1
				while ( pa != EMPTY_VOXEL && numKids[pa] == 1 ) {
					assert( rowPending[pa] );
					rowPending[pa] = false;
					numDone++;
					lookupOldRowFromNew.push_back( pa );
					pa = parentVoxel[pa];
				}
				if ( pa != EMPTY_VOXEL ) {
					assert( numKids[pa] > 0 );
					numKids[pa]--;
				}
			}
		}
	}

	cout << setprecision(4);
	cout << "oldRowFromNew= {" ;
	for ( unsigned int i = 0; i < nrows_; ++i )
		cout << lookupOldRowFromNew[i] << ", ";
	cout << "}\n";
	// Then we fill in the reordered matrix. Note we need to reorder
	// columns too.
	shuffleRows( lookupOldRowFromNew );
	return true;
}

// Fill in the reordered matrix. Note we need to reorder columns too.
void FastMatrixElim::shuffleRows( 
				const vector< unsigned int >& lookupOldRowFromNew )
{
	vector< unsigned int > lookupNewRowFromOld( nrows_ );
	for ( unsigned int i = 0; i < nrows_; ++i )
		lookupNewRowFromOld[ lookupOldRowFromNew[i] ] = i;

	FastMatrixElim temp = *this;
	clear();
	setSize( temp.nrows_, temp.nrows_ );
	for ( unsigned int i = 0; i < lookupOldRowFromNew.size(); ++i ) {
		vector< unsigned int > c;
		vector< double > e;
		unsigned int num = temp.getRow( lookupOldRowFromNew[i], e, c );
		vector< unsigned int > newc( num );
		vector< double > newe( num );
		for ( unsigned int j = 0; j < num; ++j ) {
			newc[j] = lookupNewRowFromOld[ c[j] ];
			newe[j] = e[j];
		}
		// Now we need to sort the new row entries in increasing col order.
		/*
		sortByColumn( newc, newe );
		addRow( i, newe, newc );
		*/
		sortByColumn( newc, e );
		addRow( i, e, newc );
	}
}

void sortByColumn( vector< unsigned int >& col, vector< double >& entry )
{
	unsigned int num = col.size();
	assert( num == entry.size() );
	// Stupid bubble sort, as we only have up to 5 entries and need to 
	// sort both the col and reorder the entries by the same sequence.
	for ( unsigned int i = 0; i < num; ++i ) {
		for ( unsigned int j = 1; j < num; ++j ) {
			if ( col[j] < col[j-1] ) {
				unsigned int temp = col[j];
				col[j] = col[j-1];
				col[j-1] = temp;
				double v = entry[j];
				entry[j] = entry[j-1];
				entry[j-1] = v;
			}
		}
	}
}


void FastMatrixElim::makeTestMatrix( 
				const double* test, unsigned int numCompts )
{
	setSize( numCompts, numCompts );
	vector< double > row( numCompts, ~0 );
	for ( unsigned int i = 0; i < numCompts; ++i ) {
		for ( unsigned int j = 0; j < numCompts; ++j ) {
			unsigned int k = i * numCompts + j;
			if ( test[k] < 0.1 ) {
			} else {
				N_.push_back( test[k] );
				colIndex_.push_back( j );
			}
		}
		rowStart_[i + 1] = N_.size();
	}
}

/*
I need an outer function to fill the vector of ops for forward elim.
Then I need another outer function to fill another set of ops for
back-substitution.
*/

/**
 * Builds the vector of forward ops: ratio, i, j 
 * RHS[i] = RHS[i] - RHS[j] * ratio 
 * This vec tells the routine which rows below have to be eliminated.
 * This includes the rows if any in the tridiagonal band and also 
 * rows, if any, on branches.
 */
void FastMatrixElim::buildForwardElim( vector< unsigned int >& diag,
	vector< Triplet< double > >& fops )
{
	vector< vector< unsigned int > > rowsToElim( nrows_ );
	diag.clear();
	for ( unsigned int i = 0; i < nrows_; ++i ) {
		unsigned int rs = rowStart_[i];
		unsigned int re = rowStart_[i+1];
		for ( unsigned int j = rs; j < re; ++j ) {
			unsigned int k = colIndex_[j];
			if ( k == i ) {
				diag.push_back(j);
			} else if ( k > i ) {
				rowsToElim[ i ].push_back( k );
			}
		}
	}
	for ( unsigned int i = 0; i < nrows_; ++i ) {
		double d = N_[diag[i]];
		unsigned int diagend = rowStart_[ i + 1 ];
		assert( diag[i] < diagend );
		vector< unsigned int >& elim = rowsToElim[i];
		for ( unsigned int j = 0; j < elim.size(); ++j ) {
			unsigned int erow = elim[j];
			if ( erow == i ) continue;
			unsigned int rs = rowStart_[ erow ];
			unsigned int re = rowStart_[ erow+1 ];
			// assert( colIndex_[rs] == i );
			double ratio = get( erow, i ) / d;
			// double ratio = N_[rs]/N_[diag[i]];
			for ( unsigned int k = diag[i]+1; k < diagend; ++k ) {
				unsigned int col = colIndex_[k];
				// findElimEntry, subtract it out.
				for ( unsigned int q = rs; q < re; ++q ) {
					if ( colIndex_[q] == col ) {
						N_[q] -= N_[k] * ratio;
					}
				}
			}
			fops.push_back( Triplet< double >( ratio, i, erow) );
		}
	}
	for ( unsigned int i = 0; i < rowsToElim.size(); ++i ) {
		cout << i << " :		";
		for ( unsigned int j = 0; j < rowsToElim[i].size(); ++j ) {
			cout << rowsToElim[i][j] << "	";
		}
		cout << endl;
	}
	for ( unsigned int i = 0; i < fops.size(); ++i ) {
		cout << "fops[" << i << "]=		" << fops[i].b_ << "	" << fops[i].c_ << 
				"	" << fops[i].a_ << endl;
	}
	/*
	*/
}

/** 
 * Operations to be done on the RHS for the back sub are generated and
 * put into the bops (backward ops) vector. 
 * col > row here, row is the entry being operated on, and col is given by 
 * rowsToSub.
 * offDiagVal is the value on the off-diagonal at row,col.  
 * diagVal is the value on the diagonal at [row][row].  
 * RHS[row] = ( RHS[row] - offDiagVal * RHS[col] ) / diagVal
 */
void FastMatrixElim::buildBackwardSub( vector< unsigned int >& diag,
	vector< Triplet< double > >& bops, vector< double >& diagVal )
{
	// This vec tells the routine which rows below have to be back-subbed.
	// This includes the rows if any in the tridiagonal band and also 
	// rows, if any, on branches.
	vector< vector< unsigned int > > rowsToSub( nrows_ );

	for ( unsigned int i = 0; i < nrows_; ++i ) {
		unsigned int d = diag[i] + 1;
		unsigned int re = rowStart_[i+1];
		for ( unsigned int j = d; j < re; ++j ) {
			unsigned int k = colIndex_[j];
			// At this point the row to sub is at (i, k). We need to go down
			// to the (k,k) diagonal to sub it out.
			rowsToSub[ k ].push_back( i );
		}
	}
	for ( unsigned int i = 0; i < rowsToSub.size(); ++i ) {
		cout << i << " :		";
		for ( unsigned int j = 0; j < rowsToSub[i].size(); ++j ) {
			cout << rowsToSub[i][j] << "	";
		}
		cout << endl;
	}

	diagVal.resize( 0 );
	// Fill in the diagonal terms. Here we do all entries.
	for ( unsigned int i = 0; i != nrows_ ; ++i ) {
		diagVal.push_back( 1.0 / N_[diag[i]] );
	}

	// Fill in the back-sub operations. Note we don't need to check zero.
	for ( unsigned int i = nrows_-1; i != 0 ; --i ) {
		for ( int j = rowsToSub[i].size() - 1; j != -1; --j ) {
			unsigned int k = rowsToSub[i][j];
			double val = get( k, i ); //k is the row to go, i is the diag.
			bops.push_back( Triplet< double >( val * diagVal[i], i, k ) );
		}
	}

	for ( unsigned int i = 0; i < bops.size(); ++i ) {
		cout << i << ":		" << bops[i].a_ << "	" << 
				bops[i].b_ << "	" <<  // diagonal index
				bops[i].c_ << "	" <<  // off-diagonal index
				1.0 / diagVal[bops[i].b_] << // diagonal value.
				endl;
	}
}

// Static function.
void FastMatrixElim::advance( vector< double >& y,
		const vector< Triplet< double > >& ops, // has both fops and bops.
		const vector< double >& diagVal )
{
	for ( vector< Triplet< double > >::const_iterator
				i = ops.begin(); i != ops.end(); ++i )
		y[i->c_] -= y[i->b_] * i->a_;

	assert( y.size() == diagVal.size() );
	vector< double >::iterator iy = y.begin();
	for ( vector< double >::const_iterator
				i = diagVal.begin(); i != diagVal.end(); ++i )
		*iy++ *= *i;
}

