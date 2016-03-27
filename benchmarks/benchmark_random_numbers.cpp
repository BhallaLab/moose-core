/***
 *       Filename:  benchmark_random_numbers.cpp
 *
 *    Description:  Benchmark random numbers in MOOSE.
 *
 *        Version:  0.0.1
 *        Created:  2016-03-26
 *       Revision:  none
 *
 *         Author:  Dilawar Singh <dilawars@ncbs.res.in>
 *   Organization:  NCBS Bangalore
 *
 *        License:  GNU GPL2
 */


#include <iostream>
#include <vector>
#include <random>
#include <ctime>
#include <cmath>
#include "../randnum/randnum.h"
using namespace std;

/**
 * @brief Compare MOOSE random number generator performance with c++11 mt19937
 * generator. Fixed seed.
 *
 * @param N
 */
void benchmark1( unsigned int N )
{
    cout << "N=" << N;
    clock_t startT = clock();
    for( unsigned int i = 0; i < N; ++i )
        mtrand();

    double mooseT = (float)(clock() - startT)/CLOCKS_PER_SEC/N;
    cout << ",MOOSE=" << mooseT; 

    // Using c++ library
    mt19937 rng( 0 );
    startT = clock();
    for( unsigned int i = 0; i < N; ++i )
        rng();
    double stlT = (float)(clock() - startT)/CLOCKS_PER_SEC/N;
    cout << ",STL=" << stlT;
    cout << ",STL/MOOSE=" << mooseT / stlT << endl;
}

/**
 * @brief Compare moose random number generator performance with rand()
 * function.
 *
 * @param N
 */
void benchmark2( unsigned int N)
{
    cout << "N=" << N;
    clock_t startT = clock();
    for( unsigned int i = 0; i < N; ++i )
        mtrand();
    double mooseT = (float)(clock() - startT)/CLOCKS_PER_SEC/N;
    cout << ",MOOSE=" << mooseT; 

    // Using c++ library
    mt19937 rng( 0 );
    startT = clock();
    for( unsigned int i = 0; i < N; ++i )
        rand();

    double stlT = (float)(clock() - startT)/CLOCKS_PER_SEC/N;
    cout << ",STL=" << stlT;
    cout << ",STL/MOOSE=" << mooseT / stlT << endl;
}


int main(int argc, char *argv[])
{
    cerr << "Running benchmark 1" << endl;
    for(unsigned int i = 1; i < 9; i++)
        benchmark1( pow(10,i) );
    return 0;
}
