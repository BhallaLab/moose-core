/***
 *       Filename:  LSODE.h
 *
 *    Description:  See LSODE.cpp file to more information.
 *
 *        Version:  0.0.1
 *        Created:  2018-08-09

 *       Revision:  none
 *
 *         Author:  Dilawar Singh <dilawars@ncbs.res.in>
 *   Organization:  NCBS Bangalore
 *
 *        License:  GNU GPL3
 */

#ifndef LSODE_H
#define LSODE_H

typedef void  (*_lsoda_f) (double, double *, double *, void *);

static int idamax( int n, double* dx, int incx);

void dscal(int n, double da, double* dx, int incx);

static double
ddot(int n, double* dx, int ncx, double* dy, int incy);

static void
daxpy(int n, double da, double* dx, int incx, double* dy, int incy);

static void
dgesl(double** a, int n, int* ipvt, double* b, int job);

void
dgefa(double** a, int n, int* ipvt, int* info);

void lsoda( _lsoda_f f, int neq, double *y, double *t, double tout, int itol, double *rtol, double *atol,
		   int itask, int *istate, int iopt, int jt,
		   int iwork1, int iwork2, int iwork5, int iwork6, int iwork7, int iwork8, int iwork9,
		   double rwork1, double rwork5, double rwork6, double rwork7, void *_data);


void n_lsoda_terminate(void);


#endif /* end of include guard: LSODE_H */
