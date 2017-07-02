/*
 * =====================================================================================
 *
 *       Filename:  cuda_ksolve.cpp
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

using namespace std;

__host__ __device__ void deriv(double* x0, double* array, double c, int size,  double* ki, double*
        ko);
__global__ void rk4(double* x0, double* array, double* h, int size);


__global__ void rk4(double* x0, double* array, double* h, int size){
    //int** arr = new int*[row];
    //int size = 100;//size of the species
    double* k1_v = new double [size];
    double* k2_v = new double [size];
    double* k3_v = new double [size];
    double* k4_v = new double [size];

    int tid = threadIdx.x;
    for(int i = 0; i < size; ++i){

        k1_v[i] = x0[ tid*size + i ];
        k2_v[i] = x0[ tid*size + i ];
        k3_v[i] = x0[ tid*size + i ];
        k4_v[i] = x0[ tid*size + i ];
    
    }

    deriv(x0, array, 0.0, size, k1_v, k1_v);
    deriv(x0, array, *h/2.0, size, k1_v, k2_v);
    deriv(x0, array, *h/2.0, size, k2_v, k3_v);
    deriv(x0, array, *h, size, k3_v, k4_v);

    for(int i = 0; i < size; ++i){
        x0[ tid*size + i ] = x0[ tid*size + i ] + (k1_v[i] + 2.0 *  k2_v[i] + 2.0 * k3_v[i] + k4_v[i]) *
            (*h)/6.0;  
    }
    delete[] k1_v;
    delete[] k2_v;
    delete[] k3_v;
    delete[] k4_v;
   // delete[] arr;
}


__host__ __device__ void deriv(double* x0, double* array, double c, int size,  double* ki, double* ko){

    for(int i = 0; i < size; ++i){

        for(int j = 0; j< size; ++j){

            ko[i] = ko[i] + array[ size * i + j ] * (x0[j] + c * ki[j]); 

        }

    }

}
