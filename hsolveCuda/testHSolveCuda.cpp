/**********************************************************************
 ** This program is part of 'MOOSE', the
 ** Multiscale Object Oriented Simulation Environment.
 **   copyright (C) 2003-2011 Upinder S. Bhalla, Niraj Dudani and NCBS
 ** It is made available under the terms of the
 ** GNU Lesser General Public License version 2.1
 ** See the file COPYING.LIB for the full notice.
 **********************************************************************/

#ifdef DO_UNIT_TESTS
#include "header.h"
#include "HSolveCuda.h"
#include "cudaLibrary/CudaTimer.h"

#include <stdio.h>
#include <iostream>
#include <limits>

//#include <cassert>
//#include <omp.h>

//#include <map>
//#include <vector>
#include <sstream>
#include "../shell/Shell.h"
#include "../biophysics/Compartment.h"
using namespace moose; // For moose::Compartment from 'Compartment.h'
#include "../hsolve/HSolveUtils.h"
#include "../hsolve/HSolveStruct.h"
#include "../hsolve/HinesMatrix.h"
#include "../hsolve/HSolvePassive.h"
#include "../hsolve/TestHSolve.h"

using namespace std;
/**
 * Check 2 floating-point numbers for "equality".
 * Algorithm (from Knuth) 'a' and 'b' are close if:
 *      | ( a - b ) / a | < e AND | ( a - b ) / b | < e
 * where 'e' is a small number.
 *
 * In this function, 'e' is computed as:
 * 	    e = tolerance * machine-epsilon
 */
template< class T >
bool isClose( T a, T b, T tolerance )
{
	T epsilon = std::numeric_limits< T >::epsilon();

	if ( a == b )
		return true;

	if ( a == 0 || b == 0 )
		return ( fabs( a - b ) < tolerance * epsilon );

	return (
		fabs( ( a - b ) / a ) < tolerance * epsilon
		&&
		fabs( ( a - b ) / b ) < tolerance * epsilon
	);
}

void testHSolveCuda()
{
	cout << "\n************** CUDA Test Started COUT\n";

	/*
	* Solver instance.
	*/
	const int CELL_NUMBER = 100;
	HSolveCuda hsc(CELL_NUMBER, 20);

	hsc.TestCudaAbility();

	return;
	CudaTimer tm, tmGlobal;
	tmGlobal.StartTimer();

	Shell* shell = reinterpret_cast< Shell* >( Id().eref().data() );

	vector< int* > childArray;
	vector< unsigned int > childArraySize;


	/**
	 *  We test passive-cable solver for the following cell:
	 *
	 *   Soma--->  15 - 14 - 13 - 12
	 *              |    |
	 *              |    L 11 - 10
	 *              |
	 *              L 16 - 17 - 18 - 19
	 *                      |
	 *                      L 9 - 8 - 7 - 6 - 5
	 *                      |         |
	 *                      |         L 4 - 3
	 *                      |
	 *                      L 2 - 1 - 0
	 *
	 *  The numbers are the hines indices of compartments. Compartment X is the
	 *  child of compartment Y if X is one level further away from the soma (#15)
	 *  than Y. So #17 is the parent of #'s 2, 9 and 18.
	 */

	int childArray_1[ ] =
	{
		/* c0  */  -1,
		/* c1  */  -1, 0,
		/* c2  */  -1, 1,
		/* c3  */  -1,
		/* c4  */  -1, 3,
		/* c5  */  -1,
		/* c6  */  -1, 5,
		/* c7  */  -1, 4, 6,
		/* c8  */  -1, 7,
		/* c9  */  -1, 8,
		/* c10 */  -1,
		/* c11 */  -1, 10,
		/* c12 */  -1,
		/* c13 */  -1, 12,
		/* c14 */  -1, 11, 13,
		/* c15 */  -1, 14, 16,
		/* c16 */  -1, 17,
		/* c17 */  -1, 2, 9, 18,
		/* c18 */  -1, 19,
		/* c19 */  -1,
	};

	int myArray[CELL_NUMBER][39];
	//Extra
	for (int iii = 0; iii < CELL_NUMBER; ++iii) {

		for (int index = 0; index < 39; ++index) {
			myArray[iii][index] = childArray_1[index];
		}
		childArray.push_back( myArray[iii]);
		childArraySize.push_back( sizeof( myArray[iii] ) / sizeof( int ) );
	}

	////////////////////////////////////////////////////////////////////////////
	// Run tests
	////////////////////////////////////////////////////////////////////////////

	/*
	 * This is the full reference matrix which will be compared to its sparse
	 * implementation.
	 */
	vector< vector< double > > matrix;

	/*
	 * Model details.
	 */
	double dt = 1.0;

	//Create to check with main data
	vector< TreeNodeStruct > tree;
	vector< double > Em;
	vector< double > V;

	//Temporary
	vector< vector< double > > Bs;
	Bs.resize(CELL_NUMBER);

	vector< double > VMid;


	//Time variables
	double prepShell=0, prepMattrix=0, HSUpdate=0,HSUpdateLocal=0,
			felm = 0, felmLocal = 0, bsub = 0, bsubLocal=0;
	/*
	 * Loop over cells.
	 */
	int i;
	int j;
	//~ bool success;
	int nCompt;
	int* array;
	unsigned int arraySize;


	//Generate Data for CUDA
	for ( unsigned int cell = 0; cell < childArray.size(); cell++ ) {

		array = childArray[ cell ];
		arraySize = childArraySize[ cell ];
		nCompt = count( array, array + arraySize, -1 );

		//////////////////////////////////////////
		// Prepare local information on cell
		//////////////////////////////////////////
		tree.resize( nCompt );

		for ( i = 0; i < nCompt; i++ ) {
			unsigned int index= cell*nCompt+i;
			hsc.Ra[index] = i+100;//15.0 + 3.0 * i;
			hsc.Rm[index] = 45.0 + 15.0 * i;
			hsc.Cm[index] = 500.0 + 200.0 * i * i;
			hsc.Em[index] = -0.06 ;
			hsc.V[index]= -0.06 + 0.01 * i ;

			tree[i].Ra = hsc.Ra[index];
			tree[i].Rm = hsc.Rm[index];
			tree[i].Cm = hsc.Cm[index];
			V.push_back(hsc.V[index]);
			Em.push_back(hsc.Em[index]);
		}

		int count = -1;
		for ( unsigned int a = 0; a < arraySize; a++ )
			if ( array[ a ] == -1 )
				count++;
			else
				tree[ count ].children.push_back( array[ a ] );


		//////////////////////////////////////////
		// Setup local matrix
		//////////////////////////////////////////
		//TODO: We changes the index order, make sure it is ok

		// Create local reference matrix
		makeFullMatrix(	tree, dt, matrix );

		hsc.AddFullMatrix(cell,tree,dt);

		VMid.resize( nCompt );
		Bs[cell].resize( nCompt );

		vector< vector< double > > matrixCopy;
		matrixCopy.assign( matrix.begin(), matrix.end() );

		//////////////////////////////////////////
		// Run comparisons
		//////////////////////////////////////////
		double tolerance;
//
//		/*
//		 * Compare initial matrices
//		 */
//
//		tolerance = 2.0;
//
////		for ( i = 0; i < nCompt; ++i )
////			for ( j = 0; j < nCompt; ++j ) {
////				ostringstream error;
////				error << "Testing matrix construction:"
////				      << " Cell# " << cell + 1
////				      << " A(" << i << ", " << j << ")";
////				ASSERT (
////					isClose< double >( HP.getA( i, j ), matrix[ i ][ j ], tolerance ),
////					error.str()
////				);
////			}
////
//		/*
//		 *
//		 * Gaussian elimination
//		 *
//		 */
//
//		tolerance = 4.0; // ratio to machine epsilon
////
////		for ( int pass = 0; pass < 2; pass++ ) {
//			/*
//			 * First update terms in the equation. This involves setting up the B
//			 * in Ax = B, using the latest voltage values. Also, the coefficients
//			 * stored in A have to be restored to their original values, since
//			 * the matrix is modified at the end of every pass of gaussian
//			 * elimination.
//			 */
//
//			// Do so in the solver..
////			//TIMER
////			tm.StartTimer();
////
////			HP.updateMatrix();
////
////			//TIMER
////			HSUpdate += tm.GetTimer();
////			tm.StartTimer();

////			 ..locally..
			matrix.assign( matrixCopy.begin(), matrixCopy.end() );
//
			for ( i = 0; i < nCompt; i++ )
				Bs[cell][ i ] =
					V[ i ] * tree[ i ].Cm / ( dt / 2.0 ) +
					Em[ i ] / tree[ i ].Rm;
//			printf("%f", V[0]);
//			cout << "V:" <<V[0] << "\tCm:" << tree[ 0 ].Cm << "\tdt:" <<dt<< "\tEm:" << Em[0] << "\tRm:" << tree[ 0 ].Rm << endl<<flush;

////			//TIMER
////			HSUpdateLocal += tm.GetTimer();
////
////			// ..and compare B.
////			for ( i = 0; i < nCompt; ++i ) {
////				ostringstream error;
////				error << "Updating right-hand side values:"
////				      << " Pass " << pass
////				      << " Cell# " << cell + 1
////				      << " B(" << i << ")";
////				ASSERT (
////					isClose< double >( HP.getB( i ), B[ i ], tolerance ),
////					error.str()
////				);
////			}
////
////			//TIMER
////			tm.StartTimer();
////
////			/*
////			 *  Forward elimination..
////			 */
////
////			// ..in solver..
////			HP.forwardEliminate();
////
////			//TIMER
////			felm += tm.GetTimer();
////			tm.StartTimer();
////
////			// ..and locally..
////			int k;
////			for ( i = 0; i < nCompt - 1; i++ )
////				for ( j = i + 1; j < nCompt; j++ ) {
////					double div = matrix[ j ][ i ] / matrix[ i ][ i ];
////					for ( k = 0; k < nCompt; k++ )
////						matrix[ j ][ k ] -= div * matrix[ i ][ k ];
////					B[ j ] -= div * B[ i ];
////				}
////
////			//TIMER
////			felmLocal += tm.GetTimer();
////
////			// ..then compare A..
////			for ( i = 0; i < nCompt; ++i )
////				for ( j = 0; j < nCompt; ++j ) {
////					ostringstream error;
////					error << "Forward elimination:"
////					      << " Pass " << pass
////					      << " Cell# " << cell + 1
////					      << " A(" << i << ", " << j << ")";
////					ASSERT (
////						isClose< double >( HP.getA( i, j ), matrix[ i ][ j ], tolerance ),
////						error.str()
////					);
////				}
////
////			// ..and also B.
////			for ( i = 0; i < nCompt; ++i ) {
////				ostringstream error;
////				error << "Forward elimination:"
////				      << " Pass " << pass
////				      << " Cell# " << cell + 1
////				      << " B(" << i << ")";
////				ASSERT (
////					isClose< double >( HP.getB( i ), B[ i ], tolerance ),
////					error.str()
////				);
////			}
////
////			/*
////			 *  Backward substitution..
////			 */
////
////			//TIMER
////			tm.StartTimer();
////
////			// ..in solver..
////			HP.backwardSubstitute();
////
////			//TIMER
////			bsub += tm.GetTimer();
////			tm.StartTimer();
////
////			// ..and full back-sub on local matrix equation..
////			for ( i = nCompt - 1; i >= 0; i-- ) {
////				VMid[ i ] = B[ i ];
////
////				for ( j = nCompt - 1; j > i; j-- )
////					VMid[ i ] -= VMid[ j ] * matrix[ i ][ j ];
////
////				VMid[ i ] /= matrix[ i ][ i ];
////
////				V[ i ] = 2 * VMid[ i ] - V[ i ];
////			}
////			//TIMER
////			bsubLocal += tm.GetTimer();
////
////			// ..and then compare VMid.
////			for ( i = nCompt - 1; i >= 0; i-- ) {
////				ostringstream error;
////				error << "Back substitution:"
////				      << " Pass " << pass
////				      << " Cell# " << cell + 1
////				      << " VMid(" << i << ")";
////				ASSERT (
////					isClose< double >( HP.getVMid( i ), VMid[ i ], tolerance ),
////					error.str()
////				);
////			}
////
////			for ( i = nCompt - 1; i >= 0; i-- ) {
////				ostringstream error;
////				error << "Back substitution:"
////				      << " Pass " << pass
////				      << " Cell# " << cell + 1
////				      << " V(" << i << ")";
////				ASSERT (
////					isClose< double >( HP.getV( i ), V[ i ], tolerance ),
////					error.str()
////				);
////			}
//		}
//
//		 cleanup
//		shell->doDelete( n );
	}

	//Cuda

	//Import Data to Cuda
	//TODO: Build matrix inside this method
	//TODO: Find a way to improve memory copy
	hsc.Setup(dt);
	hsc.UpdateMatrix();
	hsc.Read_B();

	// ..and compare B.
	double tolerance = 4000.0;
	for ( unsigned int cell = 0; cell < childArray.size(); cell++ ) {
		for ( i = 0; i < nCompt; ++i ) {
			cout << "diff: " << (hsc.B[cell*nCompt+i] - Bs[cell][ i ])/Bs[cell][ i ] << endl
					<< "diff: " << (hsc.B[cell*nCompt+i] - Bs[cell][ i ])/hsc.B[cell*nCompt+i] << endl
					<< flush;
			ostringstream error;
			error << "Updating right-hand side values:"
				  << " Cell# " << cell + 1<<" val= "<<hsc.B[cell*nCompt+i]
				  << ", B(" << i << ")" <<" val= "<<Bs[cell][ i ];
			ASSERT (
				isClose< double >( hsc.B[cell*nCompt+i] , Bs[cell][ i ], tolerance ),
				error.str()
			);
		}
	}
	hsc.ForwardElimination();
	hsc.BackwardSubstitute();
	//Check

//	double totalTime = tmGlobal.GetTimer();
//
//	cout << "Cell Number:			\t" << childArray.size()<<endl;
//	cout << "Compartment Number:		\t" << childArray.size()*nCompt<<endl;
//	cout << "Total&			\t" << totalTime <<"\tms"<<endl;
//	cout << "Shell Preparation &" << prepShell <<"\tms&\t"<<prepShell/totalTime*100<<" \\%\\\\"<<endl;
//	cout << "Matrix Preparation  &		\t" << prepMattrix <<"\tms&\t"<<prepMattrix/totalTime*100<<" \\%\\\\"<<endl;
//	cout << "MatrixUpdate &		\t" << HSUpdate <<"\tms&\t"<<HSUpdate/totalTime*100<<" \\%\\\\"<<endl;
//	cout << "Forward Elimination &			\t" << felm <<"\tms&\t"<<felm/totalTime*100<<" \\%\\\\"<<endl;
//	cout << "Backward Substitution &	\t" << bsub <<"\tms&\t"<<bsub/totalTime*100<<" \\%\\\\"<<endl;
//
//	cout << "Matrix Update&		\t" << HSUpdateLocal <<"\tms&\t"<<HSUpdateLocal/totalTime*100<<" \\%\\\\"<<endl;
//	cout << "Forward Elimination&		\t" << felmLocal <<"\tms&\t"<<felmLocal/totalTime*100<<" \\%\\\\"<<endl;
//	cout << "Backward Substitution&	\t" << bsubLocal <<"\tms&\t"<<bsubLocal/totalTime*100<<" \\%\\\\"<<endl;
//	cout<< flush;


	printf("\n************** CUDA Test Finished ");

}

#endif
