/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2014 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/
/*
#include <vector>
#include <algorithm>
#include <cassert>
#include <functional>
#include <iostream>
#include <iomanip>
//#include "/usr/include/gsl/gsl_linalg.h"
using namespace std;
*/
#ifdef USE_GSL
#include <gsl/gsl_linalg.h>
#endif
#include "header.h"
#include "../basecode/SparseMatrix.h"
#include "FastMatrixElim.h"
#include "../shell/Shell.h"



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
	// fe.print();
	vector< unsigned int > parentVoxel;
	vector< unsigned int > lookupOldRowsFromNew;
	parentVoxel.insert( parentVoxel.begin(), &parents[0], &parents[numCompts] );
	fe.hinesReorder( parentVoxel, lookupOldRowsFromNew );
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
	// fe.print();
	FastMatrixElim foo = fe;

	vector< unsigned int > diag;
	vector< double > diagVal;
	fe.buildForwardElim( diag, fops );
	// fe.print();
	fe.buildBackwardSub( diag, fops, diagVal );
	vector< double > y( numCompts, 1.0 );
	vector< double > ones( numCompts, 1.0 );
	FastMatrixElim::advance( y, fops, diagVal );
	/*
	for ( unsigned int i = 0; i < numCompts; ++i )
		cout << "y" << i << "]=	" << y[i] << endl;
		*/

	// Here we verify the answer
	
	vector< double > alle;
	for( unsigned int i = 0; i < numCompts; ++i ) {
		for( unsigned int j = 0; j < numCompts; ++j ) {
			alle.push_back( foo.get( i, j ) );
		}
	}
	// cout << "myCode: " << checkAns( &alle[0], numCompts, &y[0], &ones[0] ) << endl;

	assert(	checkAns( &alle[0], numCompts, &y[0], &ones[0] ) < 1e-25 );

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
		// cout << "x[" << i << "]=	" << gslAns[i] << endl;
	}
	// cout << "GSL: " << checkAns( test, numCompts, &gslAns[0], &ones[0] ) << endl;
	assert( checkAns( test, numCompts, &gslAns[0], &ones[0] ) < 1e-25 );
	gsl_vector_free( x );

	cout << "." << flush;
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

	for ( unsigned int i = 0; i < col.size(); ++i )
		assert( col[i] == (i + 1) * 10 );

	assert( entry[0] == 6 );
	assert( entry[1] == 1 );
	assert( entry[2] == 7 );
	assert( entry[3] == 2 );
	assert( entry[4] == 8 );
	assert( entry[5] == 3 );
	assert( entry[6] == 9 );
	assert( entry[7] == 4 );
	assert( entry[8] == 10 );
	assert( entry[9] == 5 );

	/*
	cout << "testing sorting\n";
	for ( unsigned int i = 0; i < col.size(); ++i ) {
		cout << "d[" << i << "]=	" << k[i] << 
		   ", col[" << i << "]= " <<	col[i] << ", e=" << entry[i] << endl;
	}
	cout << endl;
	*/
	cout << "." << flush;
}

void testSetDiffusionAndTransport()
{
	static double test[] = {
		0,  2,  0,  0,  0,  0,
		1,  0,  2,  0,  0,  0,
		0,  1,  0,  2,  0,  0,
		0,  0,  1,  0,  2,  0,
		0,  0,  0,  1,  0,  2,
		0,  0,  0,  0,  1,  0,
	};
	const unsigned int numCompts = 6;
	FastMatrixElim fm;
	fm.makeTestMatrix( test, numCompts );
	vector< unsigned int > parentVoxel( numCompts );
	parentVoxel[0] = -1;
	parentVoxel[1] = 0;
	parentVoxel[2] = 1;
	parentVoxel[3] = 2;
	parentVoxel[4] = 3;
	parentVoxel[5] = 4;

	// cout << endl;
	// fm.print();
	// cout << endl;
	// fm.printInternal();
	fm.setDiffusionAndTransport( parentVoxel, 1, 10, 0.1 );
	// cout << endl;
	// fm.print();
	// cout << endl;
	// fm.printInternal();

	for( unsigned int i =0; i < numCompts; ++i ) {
		unsigned int start = 0;
		if ( i > 0 )
			start = i - 1;
		for( unsigned int j = start ; j < i+1 && j < numCompts ; ++j ) {
			if ( i == j + 1 )
				assert( doubleEq( fm.get( i, j ), 0.1 ) );
			else if ( i + 1 == j ) {
				assert( doubleEq( fm.get( i, j ), 2.2 ) );
			} else if ( i == j ) {
				if ( i == 0 )
					assert( doubleEq( fm.get( i, j ), 0.8 ) );
				else if ( i == numCompts - 1 )
					assert( doubleEq( fm.get( i, j ), -0.1 ) );
				else 
					assert( doubleEq( fm.get( i, j ), -0.3 ) );
			}
		}
	}
	cout << "." << flush;
}

void testCylDiffn()
{
	Shell* s = reinterpret_cast< Shell* >( Id().eref().data() );
	double len = 25e-6;
	double r0 = 1e-6;
	double r1 = 1e-6;
	double diffLength = 1e-6; // 1e-6 is the highest dx for which error is OK
	double runtime = 10.0;
	double dt = 0.1; // 0.2 is the highest dt for which the error is in bounds
	// Should set explicitly, currently during creation of DiffPoolVec
	double diffConst = 1.0e-12; 
	Id model = s->doCreate( "Neutral", Id(), "model", 1 );
	Id cyl = s->doCreate( "CylMesh", model, "cyl", 1 );
	Field< double >::set( cyl, "r0", r0 );
	Field< double >::set( cyl, "r1", r1 );
	Field< double >::set( cyl, "x0", 0 );
	Field< double >::set( cyl, "x1", len );
	Field< double >::set( cyl, "lambda", diffLength );
	unsigned int ndc = Field< unsigned int >::get( cyl, "numMesh" );
	assert( ndc == static_cast< unsigned int >( round( len / diffLength )));

	Id dsolve = s->doCreate( "Dsolve", model, "dsolve", 1 );
	Field< Id >::set( dsolve, "compartment", cyl );
	// Next: build by doing reinit
	s->doUseClock( "/model/dsolve", "process", 1 );
	s->doSetClock( 1, dt );
	// Then find a way to test it.
	s->doReinit();

	vector< double > nvec = 
		LookupField< unsigned int, vector< double > >::get( 
						dsolve, "nVec", 0);
	assert( nvec.size() == ndc );
	nvec[0] = 1;
	LookupField< unsigned int, vector< double > >::set( dsolve, "nVec", 
					0, nvec);

	s->doStart( runtime );

	nvec = LookupField< unsigned int, vector< double > >::get( 
						dsolve, "nVec", 0);
	/*
	cout << endl;
	for ( unsigned int i = 0; i < nvec.size(); ++i )
		cout << nvec[i] << "	";
	cout << endl;
	*/

	double dx = diffLength;
	double err = 0.0;
	double analyticTot = 0.0;
	double myTot = 0.0;
	for ( unsigned int i = 0; i < nvec.size(); ++i ) {
		double x = i * dx + dx * 0.5;
		// This part is the solution as a func of x,t.
		double y = dx *  // This part represents the init n of 1 in dx
			( 1.0 / sqrt( PI * diffConst * runtime ) ) * 
			exp( -x * x / ( 4 * diffConst * runtime ) ); 
		err += ( y - nvec[i] ) * ( y - nvec[i] );
		//cout << i << "	" << x << "	" << y << "	" << conc[i] << endl;
		analyticTot += y;
		myTot += nvec[i];
	} 
	assert( doubleEq( myTot, 1.0 ) );
	// cout << "analyticTot= " << analyticTot << ", myTot= " << myTot << endl;
	assert( err < 1.0e-5 );


	s->doDelete( model );
	cout << "." << flush;
}

void testCellDiffn()
{
	Id makeCompt( Id parentCompt, Id parentObj,
		string name, double len, double dia, double theta );
	Shell* s = reinterpret_cast< Shell* >( Id().eref().data() );
	double len = 40e-6;
	double dia = 10e-6;
	double diffLength = 1e-6;
	double dt = 1.0e-1;
	double runtime = 1000.0;
	Id model = s->doCreate( "Neutral", Id(), "model", 1 );
	Id soma = makeCompt( Id(), model, "soma", dia, dia, 90 );
	Id dend = makeCompt( soma, model, "dend", len, 3e-6, 0 );
	Id branch1 = makeCompt( dend, model, "branch1", len, 2e-6, 45.0 );
	Id branch2 = makeCompt( dend, model, "branch2", len, 2e-6, -45.0 );
	Id twig1 = makeCompt( branch1, model, "twig1", len, 1.5e-6, 90.0 );
	Id twig2 = makeCompt( branch1, model, "twig2", len, 1.5e-6, 0.0 );

	Id nm = s->doCreate( "NeuroMesh", model, "neuromesh", 1 );
	Field< double >::set( nm, "diffLength", diffLength );
	Field< string >::set( nm, "geometryPolicy", "cylinder" );
	Field< Id >::set( nm, "cell", model );
	unsigned int ns = Field< unsigned int >::get( nm, "numSegments" );
	assert( ns == 6 );
	unsigned int ndc = Field< unsigned int >::get( nm, "numDiffCompts" );
	assert( ndc == 210  );

	Id dsolve = s->doCreate( "Dsolve", model, "dsolve", 1 );
	Field< Id >::set( dsolve, "compartment", nm );
	// Next: build by doing reinit
	s->doUseClock( "/model/dsolve", "process", 1 );
	s->doSetClock( 1, dt );
	// Then find a way to test it.
	s->doReinit();

	vector< double > nvec = 
		LookupField< unsigned int, vector< double > >::get( 
						dsolve, "nVec", 0);
	assert( nvec.size() == ndc );
	nvec[0] = 1;
	LookupField< unsigned int, vector< double > >::set( dsolve, "nVec", 
					0, nvec);

	s->doStart( runtime );

	nvec = LookupField< unsigned int, vector< double > >::get( 
						dsolve, "nVec", 0);
	double myTot = 0;
	for ( unsigned int i = 0; i < nvec.size(); ++i )
		myTot += nvec[i];
	assert( doubleEq( myTot, 1.0 ) );

	/*
	cout << endl;
	for ( unsigned int i = 0; i < nvec.size(); ++i )
		cout << nvec[i] << "	";
	cout << endl;
	*/


	s->doDelete( model );
	cout << "." << flush;
}
#if 0
void testBuildTree()
{
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
	FastMatrixElim fe;
	fe.makeTestMatrix( test, numCompts );
	fe.print();
	vector< unsigned int > parentVoxel;
	bool ret = fe.buildTree( 0, parentVoxel );
	assert( ret );
	assert( parentVoxel[0] == static_cast< unsigned int >( -1 ) );
	assert( parentVoxel[1] == 0 );
	assert( parentVoxel[2] == 0 );
	assert( parentVoxel[3] == 0 );
	assert( parentVoxel[8] == 2 );
	assert( parentVoxel[10] == 3 );
	assert( parentVoxel[5] == 4 );
	assert( parentVoxel[9] == 4 );
	assert( parentVoxel[6] == 5 );
	assert( parentVoxel[7] == 6 );

	assert( parentVoxel[10] == 2 );

	/*
	 * This is the sequence of traversal. x means empty, s means sibling,
	 * . means below diagonal. The numbers are the sequence.
	static double traverseIndex[] = {
	// col  1   2   3   4   5   6   7   8   9   10
		#,  1,  2,  3,  x,  x,  x,  x,  x,  x,  x,
		.,  #,  x,  x,  x,  x,  x,  x,  x,  x,  x,
		.,  x,  #,  s,  x,  x,  x,  x,  4,  x,  x,
		.,  x,  .,  #,  x,  x,  x,  x,  x,  x,  5,
		x,  x,  x,  x,  #,  6,  x,  x,  s,  7,  x,
		x,  x,  x,  x,  .,  #,  8,  x,  s,  x,  x,
		x,  x,  x,  x,  x,  .,  #,  9,  x,  x,  x,
		x,  x,  x,  x,  x,  x,  .,  #,  x,  x,  x,
		x,  x,  .,  x,  .,  .,  x,  x,  #,  x,  x,
		x,  x,  x,  x,  .,  x,  x,  x,  x,  #,  x,
		x,  x,  x,  .,  x,  x,  x,  x,  x,  x,  #,
	};
	*/
}
#endif

void testDiffusion()
{
	testSorting();
	testFastMatrixElim();
	testSetDiffusionAndTransport();
	testCylDiffn();
	// breaks at this point. testCellDiffn();
}
