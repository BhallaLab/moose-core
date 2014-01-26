#include <vector>
#include <algorithm>
#include <cassert>
#include <functional>
#include <iostream>
#include <gsl/gsl_linalg.h>
using namespace std;
#include "../basecode/SparseMatrix.h"

const unsigned int SM_MAX_ROWS = 200000;
const unsigned int SM_MAX_COLUMNS = 200000;
const unsigned int SM_RESERVE = 8;

class Unroll
{
	public:
		Unroll( double diag, double off, unsigned int i, unsigned int j )
			: 
				diagVal( diag ),
				offDiagVal( off ),
				row( i ),
				col( j )
		{;}
		double diagVal;
		double offDiagVal;
		unsigned int row; // On which the diagonal is located
		unsigned int col; // Col on which the offDiagVal is located.
};

class FastElim: public SparseMatrix< double >
{
	public:
		void makeTestMatrix( const double* test, unsigned int numCompts );
		/*
		void rowElim( unsigned int row1, unsigned int row2, 
						vector< double >& rhs );
						*/
		void buildForwardElim( vector< unsigned int >& diag,
				vector< Triplet< double > >& fops );
		void buildBackwardSub( vector< unsigned int >& diag,
				vector< Unroll >& bops );
		void orderBranches();
};


/*
void FastElim::rowElim( unsigned int row1, unsigned int row2, 
						vector< double >& rhs )
{
	unsigned int rs1 = rowStart_[row1];
	unsigned int rs2 = rowStart_[row2];
	unsigned int n1 = rowStart_[row1+1] - rs1;
	if ( n1 < 2 ) return;
	assert( colIndex_[rs2] == row1 );
	double diag1 = N_[diagIndex_[row1]];
	double temp = N_[rs2];
	double r = temp/diag1;

	const double* p1 = &(N_[ diagIndex_[row1] + 1 ] );
	double* p2 = N_ + 
	for ( unsigned int i = 
	N_[rs2+1]

	double* v1 = N_ + rs1;
	double* v2 = N_ + rs2;
	
	
			v2' - v1'*v2/v1

}
*/

void FastElim::makeTestMatrix( const double* test, unsigned int numCompts )
{
	setSize( numCompts, numCompts );
	vector< double > row( numCompts, ~0 );
	unsigned int i = 1;
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
void FastElim::buildForwardElim( vector< unsigned int >& diag,
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
						fops.push_back( Triplet< double >( ratio, i, col) );
					}
				}
			}
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
}

/** 
 * Operations to be done on the RHS for the back sub are generated and
 * put into the bops (backward ops) vector. 
 * col > row here, row is the entry being operated on, and col is given by 
 * rowsToSub.
 * offDiagVal is the value on the off-diagonal at row,col.  
 * diagVal is the value on the diagonal at [row][row].  
 * RHS[row] = ( RHS[row] - offDiagVal * RHS[col] ) / diagVyal
 */
void FastElim::buildBackwardSub( vector< unsigned int >& diag,
	vector< Unroll >& bops )
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
			rowsToSub[ i ].push_back( k );
		}
	}
	for ( unsigned int i = 0; i < rowsToSub.size(); ++i ) {
		cout << i << " :		";
		for ( unsigned int j = 0; j < rowsToSub[i].size(); ++j ) {
			cout << rowsToSub[i][j] << "	";
		}
		cout << endl;
	}

	// for the last entry we just want to divide by its diagonal.
	unsigned int k = nrows_ -1;
	bops.push_back( Unroll( N_[diag[ k ]], 0.0, k, k ) );
	for ( unsigned int i = nrows_-2; i != -1 ; --i ) {
		double diagval = N_[diag[i]];
		for ( unsigned int j = rowsToSub[i].size() - 1; j != -1; --j ) {
			k = rowsToSub[i][j];
			double val = get( i, k );
			bops.push_back( Unroll( diagval, val, i, k) );
		}
	}
	for ( unsigned int i = 0; i < bops.size(); ++i ) {
		cout << i << ":		" << bops[i].diagVal << "	" << 
				bops[i].offDiagVal << "	" << 
				bops[i].row <<  "	" <<
				bops[i].col <<
				endl;
	}
}

void advance( vector< double >& y,
	   const vector< Triplet< double > >& fops,
	   const vector< Unroll >& bops )
{
	for ( vector< Triplet< double > >::const_iterator
				i = fops.begin(); i != fops.end(); ++i )
		y[i->c_] -= y[i->b_] * i->a_;
	for ( vector< Unroll >::const_iterator
				i = bops.begin(); i != bops.end(); ++i )
		y[i->row] = (y[i->row] - i->offDiagVal * y[i->col] ) / i->diagVal;
}

void FastElim::orderBranches()
{
}

double checkAns( 
	const double* m, unsigned int numCompts, 
	const double* ans, const double* rhs )
{
	vector< double > check( numCompts, 0.0 );
	for ( unsigned int i = 0; i < numCompts; ++i ) {
		for ( unsigned int j = 0; j < numCompts; ++j )
			check[i] += m[i*numCompts + j] * ans[j];
	}
	double ret = 0.0;
	for ( unsigned int i = 0; i < numCompts; ++i )
		ret += (check[i] - rhs[i]) * (check[i] - rhs[i] );
	return ret;
}
main()
{
		/*
	*/
	static double test[] = {
		1,  2,  0,  0,  0,  0,  0,  0,  0,  0,  0,
		3,  4,  0,  0,  0,  0,  5,  0,  0,  0,  0,
		0,  0,  6,  7,  0,  0,  0,  0,  0,  0,  0,
		0,  0,  8,  9,  0,  0, 10,  0,  0,  0,  0,
		0,  0,  0,  0, 11, 12,  0,  0,  0,  0,  0,
		0,  0,  0,  0, 13, 14,  0,  0, 15,  0,  0,
		0, 16,  0, 17,  0,  0, 18, 19,  0,  0,  0,
		0,  0,  0,  0,  0,  0, 20, 21, 22,  0,  0,
		0,  0,  0,  0,  0, 23,  0, 24, 25, 26,  0,
		0,  0,  0,  0,  0,  0,  0,  0, 27, 28, 29,
		0,  0,  0,  0,  0,  0,  0,  0,  0, 30, 31,
	};
	const unsigned int numCompts = 11;
		/*
	static double test[] = {
		1,  2,
		3,  4
	};
	const unsigned int numCompts = 2;
	static double test[] = {
		1, 2, 0, 0,
		3, 4, 5, 0,
		0, 6, 7, 8,
		0, 0, 9, 10
	};
	const unsigned int numCompts = 4;
	static double test[] = {
		1,  2,  0,  0,  0,  0,
		3,  4,  5,  0,  0,  0,
		0,  6,  7,  8,  0,  0,
		0,  0,  9, 10, 11,  0,
		0,  0,  0, 12, 13, 14,
		0,  0,  0,  0, 15, 16,
	};
	const unsigned int numCompts = 6;
	static double test[] = {
		1,  2,  0,  0,  0,  0,
		3,  4,  0,  0,  1,  0,
		0,  0,  7,  8,  0,  0,
		0,  0,  9, 10, 11,  0,
		0,  1,  0, 12, 13, 14,
		0,  0,  0,  0, 15, 16,
	};
	const unsigned int numCompts = 6;
	*/
	FastElim fe;
	vector< Triplet< double > > fops;
	vector< Unroll > bops;
	fe.makeTestMatrix( test, numCompts );
	fe.print();
	cout << endl << endl;
	vector< unsigned int > diag;
	fe.buildForwardElim( diag, fops );
	fe.print();
	fe.buildBackwardSub( diag, bops );
	vector< double > y( numCompts, 1.0 );
	vector< double > ones( numCompts, 1.0 );
	advance( y, fops, bops );
	for ( int i = 0; i < numCompts; ++i )
		cout << "y" << i << "]=	" << y[i] << endl;

	// Here we verify the answer
	cout << "myCode: " << checkAns( test, numCompts, &y[0], &ones[0] ) << endl;

	/////////////////////////////////////////////////////////////////////
	// Here we do the gsl test.
	vector< double > temp( &test[0], &test[numCompts*numCompts] );
	gsl_matrix_view m = gsl_matrix_view_array( &temp[0], numCompts, numCompts );

	vector< double > z( numCompts, 1.0 );
	gsl_vector_view b = gsl_vector_view_array( &z[0], numCompts );
	gsl_vector* x = gsl_vector_alloc( numCompts );
	int s;
	gsl_permutation* p = gsl_permutation_alloc( numCompts );
	gsl_linalg_LU_decomp( &m.matrix, p, &s );
	gsl_linalg_LU_solve( &m.matrix, p, &b.vector, x);
	vector< double > gslAns( numCompts );
	for ( int i = 0; i < numCompts; ++i ) {
		gslAns[i] = gsl_vector_get( x, i );
		cout << "x[" << i << "]=	" << gslAns[i] << endl;
	}
	cout << "GSL: " << checkAns( test, numCompts, &gslAns[0], &ones[0] ) << endl;
	gsl_vector_free( x );
}
