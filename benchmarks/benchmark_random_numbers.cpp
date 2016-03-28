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
#include <iomanip>
#include <ctime>
#include <cmath>
#include <chrono>
#include "../randnum/randnum.h"

#define SETW "" // left  << setw(30)

#define BOOST 1
#ifdef BOOST
#include <boost/random.hpp>
#endif

using namespace std;
using namespace std::chrono;

/**
 * @brief Compare MOOSE random number generator performance with c++11 mt19937
 * generator. Fixed seed.
 *
 * @param N
 */
void benchmark_moose_vs_c11_stl_boost( unsigned int range = 10 )
{
    high_resolution_clock::time_point t1 = high_resolution_clock::now();
    high_resolution_clock::time_point t2 = high_resolution_clock::now();

    mt19937 rng( 0 );
#ifdef BOOST
    boost::random::mt19937 brng(0);
#endif

    // Before doing benchmarking, we generate few numbers to show the user that
    // everything is all right.
    for(int i = 0; i < 10; i ++ )
    {
        cout  << "MOOSE=" << mtrand() << ", STL=" << rng() 
#ifdef BOOST 
            << ",BOOST=" << brng()
#endif
            << endl;
    }

    for( unsigned int i = 0; i < range ; i ++ )
    {
        t1 = chrono::high_resolution_clock::now();
        unsigned int N = pow(10, i);
        cout << SETW << "N=" << N;

        for( unsigned int i = 0; i < N; ++i )
            mtrand();

        t2 = chrono::high_resolution_clock::now();
        double mooseT = duration_cast<duration<double>>(t2 - t1 ).count();
        cout << SETW << ",MOOSE=" << mooseT;

        // Using c++ library
        t1 = high_resolution_clock::now();
        for( unsigned int i = 0; i < N; ++i )
            rng();
        t2 = high_resolution_clock::now();
        double stlT = duration_cast<duration<double> >(t2 - t1).count();
        cout << SETW << ",STL=" << stlT;
        cout << SETW << ",MOOSE/STL=" << mooseT / stlT;

#ifdef BOOST
        t1 = high_resolution_clock::now();
        for( unsigned int i = 0; i < N; ++i )
            brng();

        t2 = high_resolution_clock::now();
        double boostT = duration_cast<duration<double> >(t2 - t1).count();
        cout << SETW << ",BOOST=" << boostT;
        cout << SETW << ",MOOSE/BOOST=" << mooseT / boostT;
#endif
        cout << endl;
    }
}

/**
 * @brief Compare moose random number generator performance with rand()
 * function.
 *
 * @param N
 */
void benchmark_moose_vs_rand( unsigned int N)
{
    //cerr << "# In this benchmark, we compare MOOSE random number generator speed " 
        //<< " with rand() function " << endl;
    cout << "N=" << N << SETW;
    clock_t startT = clock();
    for( unsigned int i = 0; i < N; ++i )
        mtrand();
    double mooseT = (float)(clock() - startT)/CLOCKS_PER_SEC/N;
    cout << ",MOOSE=" << mooseT << SETW; 

    // Using c++ library
    mt19937 rng( 0 );
    startT = clock();
    for( unsigned int i = 0; i < N; ++i )
        rand();

    double stlT = (float)(clock() - startT)/CLOCKS_PER_SEC/N;
    cout << ",STL=" << stlT << SETW;
    cout << ",STL/MOOSE=" << mooseT / stlT << SETW << endl;
}


int main(int argc, char *argv[])
{
    cerr << "Running benchmark: MOOSE vs c++11-STL" << endl;
    benchmark_moose_vs_c11_stl_boost( 10 );
    return 0;
}
