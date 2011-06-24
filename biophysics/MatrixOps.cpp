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

#define VERSION_2 1

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

void matMatMul( Matrix* A, Matrix* B, unsigned int dummy )
{
	unsigned int n = A->size();
	Matrix *C = matAlloc( n );
	dummy = 0;			//To keep the compiler happy.

	for( unsigned int i = 0; i < n; ++i )
	{
		for( unsigned int j = 0; j < n; ++j )
		{
			for( unsigned int k = 0; k < n; ++k )
				(*C)[i][j] += (*A)[i][k] * (*B)[k][j];
		}
	}

	for( unsigned int i = 0; i < n; ++i )
	{
		for( unsigned int j = 0; j < n; ++j )
			(*A)[i][j] = (*C)[i][j];
	}

	delete C;
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

/*void vecVecScalAdd( Matrix::iterator& xItr, Matrix::iterator& yItr, 
										double alpha, double beta ) 
{
	unsigned int n = (*xItr).size();	

	for( unsigned int i = 0; i < n; ++i )
		(*xItr)[i] = alpha * x[i] + beta * y[i];
}*/

double doPartialPivot( Matrix* A, unsigned int row, unsigned int col,
											 vector< unsigned int >* swaps )
{
	unsigned int n = A->size(), pivotRow = 0;
	double pivot = 0.0;		
	
	for( unsigned i = row; i < n; ++i )
	{
		if( fabs( (*A)[i][col] ) > pivot )
		{
			pivotRow = i;
			pivot = (*A)[i][col];
		}
	}

	//If pivot is non-zero, do the row swap.
	if ( pivot != 0 && pivotRow != row )
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
	else if ( pivot > 0 && pivotRow == row )	
		return (*A)[row][col];			//Choice of pivot is unchanged.
	else
		return 0;										//Matrix is singular!
}

Matrix* matInv( Matrix* A )
{
	Matrix *U, *L, *invL, *invA, *invU;		
	unsigned int n = A->size(), i, j, diagPos;
	vector < unsigned int >* swaps;
	double pivot, rowMultiplier1, rowMultiplier2;

	swaps = new vector< unsigned int >;
	U = matAlloc( n );   
	L = matAlloc( n );   

	//Creating a copy of the input matrix, as well as initializing the 
	//lower triangular matrix L.
	for (i = 0; i < n; ++i)
	{
		(*L)[i][i] = 1;
		for (j = 0; j < n; ++j)
			(*U)[i][j] = (*A)[i][j];
	}
	invL = L;

	////////////////
	//Reduction of A to upper triangular form U.
	//////////////
	diagPos = 0;
	pivot = 0;
	i = 1;
	j = 0;
	
	//Pivoting for the first column.
	pivot = doPartialPivot( U, 0, 0, swaps );

	while( diagPos < n - 1 )
	{
		//Direct computation of the terms of the inverse of the lower
		//triangular matrix L.
		(*L)[i][j] = (*U)[i][j] / (*U)[diagPos][j];

		//Carrying out the row operations in this fashion seemed like a way of 
		//reducing round-off error, at the cost of one extra floating point
		//operation.  
		rowMultiplier1 = (*U)[diagPos][j];
		rowMultiplier2 = (*U)[i][j];
		for( unsigned int k = j; k < n; ++k )
			(*U)[i][k] = ( (*U)[i][k] * rowMultiplier1 - 
									   (*U)[diagPos][k] *rowMultiplier2 ) / rowMultiplier1;

		if ( i != n - 1 ) 
			++i;
		else
		{
			++j;
			++diagPos;

			if ( diagPos < n - 1)
			{
				pivot = doPartialPivot( U, diagPos, diagPos, swaps );
				if ( doubleEq( pivot, 0.0 ) )
				{
					cerr << "Matrix is singular!\n";
					return 0;
				}
			}

			i = diagPos + 1;	
		}
	}
	//End of computation of L and U.
	////////////////////////////

	////////////////////////////
	//Obtaining the inverse of U and L, which is obtained by solving the 
	//simple systems Ux = I and Lx= I.
	///////////////////////////

	invU = U;
	double sum = 0;
	int k,l,m;
	
	//We serially solve for the equations Ux = e_n, Ux=e_{n-1} ..., Ux = e1. 
	//where, e_k is the k'th elementary basis vector.
	for( k = n - 1; k >= 0; --k )
	{
		for ( l = k; l >= 0; --l )
		{
			m = k;
			sum = 0;
			while ( m > l && l != k)
			{
				sum += (*U)[l][m] * (*invU)[m][k];
				--m;
			}

			if ( l == k )
				(*invU)[l][k] = (1 - sum)/(*U)[l][l]; 
			else
				(*invU)[l][k] = -sum/(*U)[l][l];
		}
	}

	//Similarly as above, we find the inverse of the lower triangular matrix by
	//back-substitution.

	invL = L;
	for( k = 0; k <= n - 1; ++k )
	{
		for( l = k + 1; l <= n - 1; ++l )  
		{
			m = 0;
			sum = 0;
			while( m < l ) 
			{
				sum += (*L)[l][m] * (*invL)[m][k];
				++m;
			}

			(*invL)[l][k] = -sum;
		}
	}
	//End of computation of invL and invU. Note that they have been computed in
	//place, which means the original copies of L and U are now gone.
	/////////////////////////////
	
	/////////////////////////////
	//Constructing the inverse of the permutation matrix P. 
	//P is calculated only if there was any pivoting carried out.
	////////////////////////////

	matMatMul( invU, invL, VERSION_2 );	
	if ( !swaps->empty() )
	{
		Matrix *P;

		P = matAlloc( n );

		for( unsigned int k = 0; k < n; ++k )
			(*P)[k][k] = 1;

		for( unsigned int k = 0; k < swaps->size(); ++k )
		{	
			i = (*swaps)[k] % 10;
			j = ( (*swaps)[k] / 10 ) % 10;

			Matrix::iterator row1Itr = (P->begin() + i); 
			Matrix::iterator row2Itr = (P->begin() + j); 
			swap( *row1Itr, *row2Itr );
		}
		matMatMul( invU, P, VERSION_2 );

		delete P;
	}

	////////////////////////
	//At this stage, L^(-1) has already been calculated. U^(-1) is just -U, and
	//P^(-1) = P^T. 
	//Since we have computed PA = LU, 
	//the final inverse is given by U^(-1)*L^(-1)*P^(-1).	
	//If P was not calculated i.e. there were no exchanges, then the 
	//inverse is just U^(-1) * L^(-1).
	////////////////////////

	invA = invU;

	delete invL;
	delete swaps;
	//Cannot delete U or invU as invA points to them!

	return invA;
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
