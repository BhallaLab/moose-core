/*
 * =====================================================================================
 *
 *       Filename:  cuda_Ksolve.h
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  Sunday 02 July 2017 12:03:59  IST
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Micky Droch (), mickydroch@gmail.com
 *   Organization:  IIT Bombay
 *
 * =====================================================================================
 */

#ifndef CUDA_KSOLVE_H

#define CUDA_KSOLVE_H

__host__ __device__ 
void deriv(double* x0, double* array, double c, int size,  double* ki, double* ko);

__global__ 
void rk4(double* x0, double* array, double* h, int size);


#endif /* end of include guard: CUDA_KSOLVE_H */
