/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2011 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _MATRIXOPS_H
#define _MATRIXOPS_H
/////////////////////////////
//Few functions for performing simple matrix operations.
//Note that all matrices here are square, which is the type encountered in
//solving the differential equations associated with first-order kinetics of
//the Markov channel model. 
//
//Author : Vishaka Datta S, 2011, NCBS
////////////////////////////
using std::vector;

typedef vector< vector< double > > Matrix;
typedef vector< double > Vector;

//Computes product of two square matrices.  
//Version 1 : Returns the result in a new matrix.
Matrix* matMatMul( Matrix*, Matrix* ); 

//Version 2 : Returns the result in the first matrix.
//The third parameter is there merely to differentiate it from the first one. 
void matMatMul( Matrix*, Matrix*, unsigned int );

//Computes sum of two square matrices.
Matrix* matMatAdd( Matrix&, Matrix& );

//Computes difference of two square matrices.
Matrix* matMatSub( Matrix&, Matrix& );

//Computes the result of multiplying all terms of a matrix by a scalar.
Matrix* matScalMul( Matrix&, double );

//Computes the result of multiplying a row vector with a matrix (in that order).
Vector* vecMatMul( Vector&, Matrix& );

//Computes the result of multiplying a matrix with a column vector (in that order).
Vector* matVecMul( Matrix&, Vector& );

//Computes the result of alpha*x + beta*y, where x and y are vectors. This is a 
//helper function for the matInv function. The result is stored in the 
//first argument.  
//void vecVecScalAdd( Matrix::iterator&, Matrix::iterator&, double, double );

//Trace of matrix.
double matTrace( Matrix& );

//Calculate column norm of matrix.
double matColNorm( Matrix& );

//Plain old matrix transpose i.e. done out-of-place.
Matrix* matTrans( Matrix* ); 

//Matrix inverse implemented using LU-decomposition (Doolittle algorithm)
//Returns NULL if matrix is singular.  
Matrix* matInv( Matrix* );

//Carry out partial pivoting. 
double doPartialPivot( Matrix*, unsigned int, unsigned int, vector< unsigned int >* );
/////////
//Memory allocation routines.
////////
Matrix* matAlloc( unsigned int );

Vector* vecAlloc( unsigned int );

#endif
