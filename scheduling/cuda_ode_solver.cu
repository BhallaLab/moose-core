/*
 * =====================================================================================
 *
 *       Filename:  cuda_ode_solver.cu
 *
 *    Description: Contains "RK4 METHOD" Code 
 *
 *        Version:  1.0
 *        Created:  Sunday 02 July 2017 11:49:03  IST
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Micky Droch (), mickydroch@gmail.com
 *   Organization:  IIT Bombay
 *
 * =====================================================================================
 */

#include <iostream>
#include <stdio.h>

#include "cuda_ode_solver.h"

using namespace std;


__global__ void rk4(double* x0, double* array, double* h, int* size){


    //int** arr = new int*[row];
    //int size = 100;//size of the species
    //int size = sizee[0];
//    int size = *sizee;

    double* k1_v = new double [*size];
    double* k2_v = new double [*size];
    double* k3_v = new double [*size];
    double* k4_v = new double [*size];

    int tid = threadIdx.x;
    for(int ix = 0; ix < (*size); ++ix){

        k1_v[ix] = x0[ tid*(*size) + ix ];
        k2_v[ix] = x0[ tid*(*size) + ix ];
        k3_v[ix] = x0[ tid*(*size) + ix ];
        k4_v[ix] = x0[ tid*(*size) + ix ];
    
    }

    deriv(x0, array, 0.0, *size, k1_v, k1_v);
    deriv(x0, array, *h/2.0, *size, k1_v, k2_v);
    deriv(x0, array, *h/2.0, *size, k2_v, k3_v);
    deriv(x0, array, *h, *size, k3_v, k4_v);

    for(int ix = 0; ix < (*size); ++ix){
        x0[ tid*(*size) + ix ] = x0[ tid*(*size) + ix ] + (k1_v[ix] + 2.0 *  k2_v[ix]
                + 2.0 * k3_v[ix] + k4_v[ix]) *
            (*h)/6.0;  
    }
    delete[] k1_v;
    delete[] k2_v;
    delete[] k3_v;
    delete[] k4_v;
   // delete[] arr;
}


__host__ __device__ void deriv(double* x0, double* array, double c, int size,  double* ki, double* ko){

    for(int ix = 0; ix < size; ++ix){

        for(int jx = 0; jx < size; ++jx){

            ko[ix] = ko[ix] + array[ size * ix + jx ] * (x0[jx] + c * ki[jx]); 

        }

    }

}
