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



void testFastMatrixElim()
{


/*
2    11
 1  4
   3    10
    9  5
     6
     7
     8

        1 2 3 4 5 6 7 8 9 10 11
1       x x x x
2       x x          
3       x   x x         x
4       x   x x              x
5               x x     x x
6               x x x   x
7                 x x x
8                   x x  
9           x   x x     x  
10              x         x   
11            x              x
	static double test[] = {
		1,  2,  3,  4,  0,  0,  0,  0,  0,  0,  0,
		5,  6,  0,  0,  0,  0,  0,  0,  0,  0,  0,
		7,  0,  8,  9,  0,  0,  0,  0, 10,  0,  0,
		11, 0, 12, 13,  0,  0,  0,  0,  0,  0, 14,
		0,  0,  0,  0, 15, 16,  0,  0, 17, 18,  0,
		0,  0,  0,  0, 19, 20, 21,  0, 22,  0,  0,
		0,  0,  0,  0,  0, 23, 24, 25,  0,  0,  0,
		0,  0,  0,  0,  0,  0, 26, 27,  0,  0,  0,
		0,  0, 28,  0, 29, 30,  0,  0, 31,  0,  0,
		0,  0,  0,  0, 32,  0,  0,  0,  0, 33,  0,
		0,  0,  0, 34,  0,  0,  0,  0,  0,  0, 35,
	};
	const unsigned int numCompts = 11;
//	static unsigned int parents[] = { 3,1,9,3,6,7,8,-1,6,5,4 };
	static unsigned int parents[] = { 2,0,8,2,5,6,7,-1,5,4,3 };
*/

/*
1   3
 2 4
   7   5
    8 6
     9
     10
     11

        1 2 3 4 5 6 7 8 9 10 11
1       x x
2       x x         x
3           x x
4           x x     x
5               x x
6               x x     x
7         x   x     x x
8                   x x x
9                 x   x x x
10                      x x  x
11                        x  x
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
	static unsigned int parents[] = { 1,6,3,6,5,8,7,8,9,10,-1};
*/

/*
Linear cable, 12 segments.
*/

	static double test[] = {
		1,  2,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
		3,  4,  5,  0,  0,  0,  0,  0,  0,  0,  0,  0,
		0,  6,  7,  8,  0,  0,  0,  0,  0,  0,  0,  0,
		0,  0,  9, 10, 11,  0,  0,  0,  0,  0,  0,  0,
		0,  0,  0, 12, 13, 14,  0,  0,  0,  0,  0,  0,
		0,  0,  0,  0, 15, 16, 17,  0,  0,  0,  0,  0,
		0,  0,  0,  0,  0, 18, 19, 20,  0,  0,  0,  0,
		0,  0,  0,  0,  0,  0, 21, 22, 23,  0,  0,  0,
		0,  0,  0,  0,  0,  0,  0, 24, 25, 26,  0,  0,
		0,  0,  0,  0,  0,  0,  0,  0, 27, 28, 29,  0,
		0,  0,  0,  0,  0,  0,  0,  0,  0, 30, 31, 32,
		0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 33, 34,
	};
	const unsigned int numCompts = 12;
	static unsigned int parents[] = { 1,2,3,4,5,6,7,8,9,10,11, unsigned(-1)};

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
	// testSorting(); // seems to work fine.
	FastMatrixElim fe;
	vector< Triplet< double > > fops;
	fe.makeTestMatrix( test, numCompts );
	fe.print();
	cout << endl << endl;
	vector< unsigned int > parentVoxel;
	parentVoxel.insert( parentVoxel.begin(), &parents[0], &parents[numCompts] );
	fe.hinesReorder( parentVoxel );
	/*
	*/
	/*
	vector< unsigned int > shuf;
	for ( unsigned int i = 0; i < numCompts; ++i )
		shuf.push_back( i );
	shuf[0] = 1;
	shuf[1] = 0;
	fe.shuffleRows( shuf );
	*/
	fe.print();
	cout << endl << endl;
	FastMatrixElim foo = fe;

	vector< unsigned int > diag;
	vector< double > diagVal;
	fe.buildForwardElim( diag, fops );
	fe.print();
	fe.buildBackwardSub( diag, fops, diagVal );
	vector< double > y( numCompts, 1.0 );
	vector< double > ones( numCompts, 1.0 );
	FastMatrixElim::advance( y, fops, diagVal );
	for ( unsigned int i = 0; i < numCompts; ++i )
		cout << "y" << i << "]=	" << y[i] << endl;

	// Here we verify the answer
	
	vector< double > alle;
	for( unsigned int i = 0; i < numCompts; ++i ) {
		for( unsigned int j = 0; j < numCompts; ++j ) {
			alle.push_back( foo.get( i, j ) );
		}
	}
	cout << "myCode: " << 
			checkAns( &alle[0], numCompts, &y[0], &ones[0] ) << endl;



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
	for ( unsigned int i = 0; i < numCompts; ++i ) {
		gslAns[i] = gsl_vector_get( x, i );
		cout << "x[" << i << "]=	" << gslAns[i] << endl;
	}
	/*
	*/
	cout << "GSL: " << checkAns( test, numCompts, &gslAns[0], &ones[0] ) << endl;
	gsl_vector_free( x );


}

void testSorting()
{
	static unsigned int k[] = {20,40,60,80,100,10,30,50,70,90};
	static double d[] = {1,2,3,4,5,6,7,8,9,10};
	vector< unsigned int > col;
	col.insert( col.begin(), k, k+10);
	vector< double > entry;
	entry.insert( entry.begin(), d, d+10);
	sortByColumn( col, entry );
	cout << "testing sorting\n";
	for ( unsigned int i = 0; i < col.size(); ++i ) {
		cout << "d[" << i << "]=	" << k[i] << 
		   ", col[" << i << "]= " <<	col[i] << ", e=" << entry[i] << endl;
	}
	cout << endl;
}

void testDiffusion()
{
	testSorting();
	testFastMatrixElim();
}
