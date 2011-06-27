/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2011 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/
#include <vector>
#include <math.h>
#include "doubleEq.h"
#include <iostream>
#include "MatrixOps.h" 

#define FIRST 1
#define SECOND 2

//Idea taken from the implementation of the DGETRF method in LAPACK. When 
//the pivot is zero, we divide by a small number instead of simply throwing up
//an error and not returning a result.
#define EPSILON 1e-15

using std::cerr;

Matrix* matMatMul( Matrix* A, Matrix* B )
{
	unsigned int n = A->size();
	Matrix *C = matAlloc( n );

	for( unsigned int i = 0; i < n; ++i )
	{
		for( unsigned int j = 0; j < n; ++j )
		{
			for( unsigned int k = 0; k < n; ++k )
				(*C)[i][j] += (*A)[i][k] * (*B)[k][j];
		}
	}

	return C;
}

void matMatMul( Matrix* A, Matrix* B, unsigned int resIndex )
{
	unsigned int n = A->size();
	Matrix *C = matAlloc( n );
//	dummy = 0;			//To keep the compiler happy.

	for( unsigned int i = 0; i < n; ++i )
	{
		for( unsigned int j = 0; j < n; ++j )
		{
			for( unsigned int k = 0; k < n; ++k )
				(*C)[i][j] += (*A)[i][k] * (*B)[k][j];
		}
	}

	if ( resIndex == FIRST )
		*A = *C;
	else if ( resIndex == SECOND )
		*B = *C;
}

void matPermMul( Matrix* A, vector< unsigned int >* swaps )
{
	unsigned int i,j,n,index;
	double temp;
	n = A->size();

	while (!swaps->empty())
	{
		index = swaps->back();
		swaps->pop_back();
		i = index % 10;
		j = (index / 10 ) % 10;

		for( unsigned int l = 0; l < n; ++l ) 
		{
			temp = (*A)[l][i];
			(*A)[l][i] = (*A)[l][j];
			(*A)[l][j] = temp;
		}
	}
}

Matrix* matMatAdd( Matrix& A, Matrix& B ) 
{
	unsigned int n = A.size();
	Matrix *C = matAlloc( n );

	for( unsigned int i = 0; i < n; ++i )
	{
		for( unsigned int j = 0; j < n; ++j )
			(*C)[i][j] = A[i][j] + B[i][j];
	}

	return C;
}

Matrix* matMatSub( Matrix& A, Matrix& B ) 
{
	unsigned int n = A.size();
	Matrix *C = matAlloc( n );

	for( unsigned int i = 0; i < n; ++i )
	{
		for( unsigned int j = 0; j < n; ++j )
			(*C)[i][j] = A[i][j] - B[i][j];
	}

	return C;
}

Matrix* matScalMul( Matrix& A, double k)
{
	unsigned int n = A.size();
	Matrix *C = matAlloc( n );

	for( unsigned int i = 0; i < n; ++i )
	{
		for( unsigned int j = 0; j < n; ++j )
			(*C)[i][j] = A[i][j] * k;
	}

	return C;
}

Vector* vecMatMul( Vector& v, Matrix& A )
{
	unsigned int n = A.size();
	Vector* w = vecAlloc( n );

	for( unsigned int i = 0; i < n; ++i )
	{
		for( unsigned int j = 0; j < n; ++j )
			(*w)[i] += v[j] * A[j][i];
	}

	return w;
}

Vector* matVecMul( Matrix& A, Vector& v )
{
	unsigned int n = A.size();
	Vector* w = vecAlloc( n );

	for( unsigned int i = 0; i < n; ++i )
	{
		for( unsigned int j = 0; j < n; ++j )
			(*w)[i] += A[i][j] * v[j];
	}

	return w;
}

double matTrace( Matrix& A )
{
	unsigned int n = A.size();
	double trace = 0;

	for ( unsigned int i = 0; i < n; ++i )
		trace += A[i][i];

	return trace;	
}

double matColNorm( Matrix& A )
{
	double norm = 0, colSum = 0;
	unsigned int n = A.size();

	for( unsigned int i = 0; i < n; ++i )
	{
		colSum = 0;
		for( unsigned int j = 0; j < n; ++j )
			colSum += fabs( A[j][i] );	
		
		if ( colSum > norm )
			norm = colSum;
	}

	return norm;
}

Matrix* matTrans( Matrix* A )
{
	unsigned int n = A->size();
	Matrix* At = matAlloc( n );

	for( unsigned int i = 0; i < n; ++i )
	{
		for( unsigned int j = 0; j < n; ++j )
			(*At)[i][j] = (*A)[j][i];
	}

	return At;
}

double doPartialPivot( Matrix* A, unsigned int row, unsigned int col,
											 vector< unsigned int >* swaps )
{
	unsigned int n = A->size(), pivotRow = row;
	double pivot = (*A)[row][col];		
	
	for( unsigned i = row; i < n; ++i )
	{
		if( fabs( (*A)[i][col] ) > pivot )
		{
			pivotRow = i;
			pivot = (*A)[i][col];
		}
	}

	//If pivot is non-zero, do the row swap.
	if ( !doubleEq(pivot,0.0) && pivotRow != row )
	{
		Matrix::iterator pivotRowItr = (A->begin() + pivotRow); 
		Matrix::iterator currRowItr = (A->begin() + row); 
		swap( *pivotRowItr, *currRowItr );

		//The row numbers interchanged are stored as a 2-digit number and pushed
		//into this vector. This information is later used in creating the 
		//permutation matrices.
		swaps->push_back( 10 * pivotRow + row ); 
		return pivot;
	}
	else if ( !doubleEq(pivot, 0.0) && pivotRow == row )	
		return (*A)[row][col];			//Choice of pivot is unchanged.
	else
		return 0;										//Matrix is singular!
}

void matInv( Matrix* A, vector< unsigned int >* swaps, Matrix* invA )
{
	Matrix *L, *invL;		
	unsigned int n = A->size(), i, j, diagPos;
	double pivot, rowMultiplier1, rowMultiplier2;

	L = matAlloc( n );   
	invL = matAlloc( n );
	*invA = *A;

	//The upper triangular portion is stored and inverted in invA

	//Creating a copy of the input matrix, as well as initializing the 
	//lower triangular matrix L.
	for (i = 0; i < n; ++i)
		(*L)[i][i] = 1;

	////////////////////////////////////
	//Calculation of LU decomposition.
	///////////////////////////////////
	for ( unsigned int k = 0; k < n; ++k )
		pivot = doPartialPivot( invA, k, k, swaps );

	diagPos = 0;
	pivot = 0;
	i = 1;
	j = 0;
	while( diagPos < n - 1 )
	{

		//Carrying out the row operations in this fashion seemed like a way of 
		//reducing round-off error, at the cost of one extra floating point
		//operation.  
		rowMultiplier1 = (*invA)[diagPos][j];		//Pivot element.
		rowMultiplier2 = (*invA)[i][j];

		for( unsigned int k = j; k < n; ++k )
			(*invA)[i][k] = ( (*invA)[i][k] * rowMultiplier1 - 
									   (*invA)[diagPos][k] *rowMultiplier2 ) / rowMultiplier1;

		(*L)[i][j] = rowMultiplier2 / rowMultiplier1;

		if ( i != n - 1 ) 
			++i;
		else
		{
			++j;
			++diagPos;
			//In case of zero pivot, divide by a very small number whose value we
			//know.
			if ( doubleEq( (*invA)[diagPos][diagPos], 0.0 ) )
			{
				cerr << "Warning : Singularity detected. Proceeding with computation"
								"anyway.\n";
				(*invA)[diagPos][diagPos] = EPSILON; 
			}

			i = diagPos + 1;	
		}
	}

/*	if ( doubleEq( (*invA)[n-1][n-1], 0.0 ) ) 
		cerr << "Warning : Matrix is singular!\n";*/

	//End of computation of L and U (which is stored in invA).
	////////////////////////////

	////////////////////////////
	//Obtaining the inverse of invA and L, which is obtained by solving the 
	//simple systems Ux = I and Lx= I.
	///////////////////////////

	double sum = 0;

	//We serially solve for the equations Ux = e_n, Ux=e_{n-1} ..., Ux = e1. 
	//where, e_k is the k'th elementary basis vector.
	for( int k = n - 1; k >= 0; --k )
	{
		for ( int l = k; l >= 0; --l )
		{
			sum = 0;

			if ( l != k )
			{
				for ( int m = k; m > l; --m )
					sum += (*invA)[l][m] * (*invA)[m][k];
			}

			if ( l == k )
				(*invA)[l][k] = (1 - sum)/(*invA)[l][l]; 
			else
				(*invA)[l][k] = -sum/(*invA)[l][l];
		}
	}

	//Similarly as above, we find the inverse of the lower triangular matrix by
	//forward-substitution.

	*invL = *L;
	double invTemp;

	//Always true when the diagonal elements are 1.
	for( unsigned int k = 0; k < n - 1; ++k )
		(*invL)[k+1][k] = -(*invL)[k+1][k];

	for( unsigned int k = 0; k <= n - 1; ++k )
	{
		for( unsigned int l = k + 2; l <= n - 1; ++l )  
		{
			invTemp = 0;
			for ( unsigned int m = k + 1; m <= n - 1; ++m )
				invTemp -= (*invL)[m][k] * (*L)[l][m];

			(*invL)[l][k] = invTemp;
		}
	}


	//End of computation of invL and invU. Note that they have been computed in
	//place, which means the original copies of L and U are now gone.
	/////////////////////////////
	
	/////////////////////////////
	//Constructing the inverse of the permutation matrix P. 
	//P is calculated only if there was any pivoting carried out.
	////////////////////////////

	matMatMul( invA, invL, FIRST );	
	if ( !swaps->empty() )
		matPermMul( invA, swaps );

	////////////////////////
	//At this stage, L^(-1) has already been calculated. U^(-1) is just -U, and
	//P^(-1) = P^T. 
	//Since we have computed PA = LU, 
	//the final inverse is given by U^(-1)*L^(-1)*P^(-1).	
	//If P was not calculated i.e. there were no exchanges, then the 
	//inverse is just U^(-1) * L^(-1).
	////////////////////////

	delete invL;
	delete L;

}

Matrix* matAlloc( unsigned int n )
{
	Matrix* A = new Matrix;

	A->resize( n );
	for ( unsigned int i = 0; i < n; ++i )
		(*A)[i].resize( n );

	return A;
}

Vector* vecAlloc( unsigned int n )
{
	Vector* vec = new Vector;

	vec->resize( n );

	return vec;
}	
