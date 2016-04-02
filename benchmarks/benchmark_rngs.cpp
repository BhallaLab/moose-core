/***
 *       Filename:  benchmark_random_numbers.cpp
 *
 *    Description:  Benchmark random numbers generator. Compare MOOSE randnum
 *    generator speed with STL-c11 and BOOST implementation.
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
#include <array>
#include <random>
#include <iomanip>
#include <ctime>
#include <cmath>
#include <fstream>
#include <chrono>
#include "../randnum/randnum.h"
#include "genrand_nishimura_masumoto.h"

#ifdef USE_GSL
#include <gsl/gsl_rng.h>
#endif


#define SETW "" // left  << setw(30)

#define MOOSE 1
#define BOOST 1
#ifdef BOOST
#include <boost/random.hpp>
#endif

using namespace std;
using namespace std::chrono;

void print_array( const vector<int>& arr, unsigned int n)
{
    for( unsigned int i = 0; i < n ; i++)
        cerr << " " << arr[i];
}

/**
 * @brief This function does the sanity check of random number generated. 
 */
void generate_sample( void )
{
    size_t N = pow(10,  6);
    vector<size_t> mooseVec(N), stlVec(N), tnmmVec(N);
    mt19937 rng( 0 );

#ifdef BOOST
    vector<size_t> boostVec(N);
    boost::random::mt19937 brng(0);
#endif

#ifdef USE_GSL 
    vector<size_t> gslVec(N);
    gsl_rng *r;
    const gsl_rng_type * T;
    gsl_rng_env_setup(); // b default it assumes mt19937 with seed 0.
    T = gsl_rng_default;
    r = gsl_rng_alloc( T );
#endif

    size_t init_arr[4]={0x123, 0x234, 0x345, 0x456}, length=4;
    tnmm::init_by_array(init_arr, length);

    for (unsigned int i = 0; i < N; ++i) 
    {
        mooseVec[i] = genrand_int32();
        stlVec[i] = rng();
#ifdef BOOST
        boostVec[i] = brng();
#endif
#ifdef USE_GSL 
        gslVec[i] = gsl_rng_get( r );
#endif
        tnmmVec[i] = tnmm::genrand_int32();
    }

    // Once numbers are collected, let's do some sanity test on random numbers
    // generated.
    ofstream outF;
    outF.open("sample_numbers.csv");
    outF << "# MOOSE, STL";
#ifdef BOOST
    outF << ",BOOST";
#endif
#ifdef USE_GSL 
    outF << ",GSL";
#endif
    outF << ",TNMM \n";
    for (unsigned int i = 0; i < N; ++i) 
    {
        outF << mooseVec[i] << "," << stlVec[i] ;
#ifdef BOOST
        outF << "," << boostVec[i];
#endif
#ifdef USE_GSL 
        outF << "," << gslVec[i];
#endif
        outF << "," << tnmmVec[i] ;
        outF << endl;
    }

    cerr << "Wrote " << N << " sampled to sample_numbers.csv file" << endl;
    outF.close();
}

/**
 * @brief Compare MOOSE random number generator performance with c++11 mt19937
 * generator. Fixed seed.
 *
 * @param N
 */
void benchmark_moose_tnmm_boost_vs_stl( unsigned int start, unsigned int stop )
{
    ofstream dataF;
    dataF.open( "./compare_rngs.csv" , ios_base::app );

    dataF << "# N,STL";
#ifdef MOOSE
    dataF << ",MOOSE,MOOSE/STL";
#endif
#ifdef BOOST
    dataF  << ",BOOST,BOOST/STL";
#endif
#ifdef USE_GSL
    dataF << ",GSL,GSL/STL";
#endif
    dataF << ",TNMM,TNMM/STL" << endl;

    high_resolution_clock::time_point t1 = high_resolution_clock::now();
    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    mt19937 rng( 0 );
    size_t init_arr[4]={0x123, 0x234, 0x345, 0x456}, length=4;
    tnmm::init_by_array(init_arr, length);

#ifdef BOOST
    boost::random::mt19937 brng(0);
#endif

#ifdef USE_GSL
    gsl_rng *r;
    const gsl_rng_type * T;
    gsl_rng_env_setup(); // b default it assumes mt19937 with seed 0.
    T = gsl_rng_default;
    r = gsl_rng_alloc( T );
    cerr << "GSL random number generator: " << gsl_rng_name( r ) << endl;
#endif

    // Before doing benchmarking, we generate few numbers to show the user that
    // everything is all right.
    for(int i = 0; i < 10; i ++ )
    {
        cerr << "STL=" << rng();
#ifdef MOOSE
        cerr  << ",MOOSE=" << genrand_int32(); 
#endif

#ifdef BOOST 
        cerr << ",BOOST=" << brng();
#endif
#ifdef USE_GSL
        cerr << ",GSL=" << gsl_rng_get( r );
#endif
        cerr << ",TNMM=" << tnmm::genrand_int32() << endl;
    }


    for( unsigned int i = start; i < stop ; i ++ )
    {
        size_t  N = pow(10, i);
        vector<int> store(N);

        dataF << SETW << N;
        double baselineT = 0.0;

        cout << "N = " << N << endl;

        cout  << "\tBaseline (vector storage time) ";
        t1 = chrono::high_resolution_clock::now();
        for(size_t i = 0; i < N; i ++)
            store[i] = 1222; // just store some number.
        t2 = chrono::high_resolution_clock::now();
        baselineT = duration_cast<duration<double>>(t2 - t1 ).count();
        cout << "  ends. Time " << baselineT << endl;

        // Using c++ library
        cout << "\tSTL starts .. ";
        t1 = high_resolution_clock::now();
        for( unsigned int i = 0; i < N; ++i )
            store[i] = rng();
        t2 = high_resolution_clock::now();

        // Get at least some numbers generated for sanity test.
        print_array( store, 10);

        double stlT = duration_cast<duration<double> >(t2 - t1).count();
        cout << " ends. Time " << stlT - baselineT << endl;
        dataF << "," << stlT - baselineT;


#ifdef MOOSE
        cout << "\tMOOSE start.. ";
        t1 = chrono::high_resolution_clock::now();
        for( size_t i = 0; i < N; ++i )
            store[i] = mtrand();
        t2 = chrono::high_resolution_clock::now();

        // Get at least some numbers generated for sanity test.
        print_array( store, 10);
        double mooseT = duration_cast<duration<double>>(t2 - t1 ).count();
        cout << "  ends. Time " << mooseT - baselineT << endl;
        dataF << "," << mooseT - baselineT;
        dataF << "," << 1.0 / (mooseT - baselineT) * (stlT - baselineT);
#endif

#ifdef BOOST
        t1 = high_resolution_clock::now();
        cout << "\tBOOST starts ..";
        for( size_t i = 0; i < N; ++i )
            store[i] = brng();
        t2 = high_resolution_clock::now();

        // Get at least some numbers generated for sanity test.
        print_array( store, 10);

        double boostT = duration_cast<duration<double> >(t2 - t1).count();
        cout << "  ends. Time " <<  boostT - baselineT << endl;
        dataF << "," << boostT - baselineT;
        dataF << "," << 1 / (boostT - baselineT) * (stlT - baselineT);
#endif

#ifdef USE_GSL 
        t1 = high_resolution_clock::now();
        cout << "\tGSL starts ..";
        for( unsigned int i =0; i < N; ++i)
            store[i] = gsl_rng_get( r );
        t2 = high_resolution_clock::now();

        // Get at least some numbers generated for sanity test.
        print_array( store, 10);

        double gslT = duration_cast<duration<double> >(t2 - t1).count();
        cout << "  ends. Time " <<  gslT - baselineT << endl;
        dataF << "," << gslT - baselineT;
        dataF << "," << 1.0 / (gslT - baselineT) * (stlT - baselineT);
#endif
        t1 = high_resolution_clock::now();
        cout << "\t TNMM starts ..";
        for( unsigned int i =0; i < N; ++i)
            store[i] = tnmm::genrand_int32( );
        t2 = high_resolution_clock::now();

        print_array(store, 10);

        double tnmmT = duration_cast<duration<double> >(t2 - t1).count();
        cout << "  ends. Time " << tnmmT - baselineT << endl;
        dataF << "," << tnmmT - baselineT;
        dataF << "," << 1/(tnmmT - baselineT) * (stlT - baselineT);
        dataF << endl;
    }
    dataF.close();
}


int main(int argc, char *argv[])
{
    cerr << "Running benchmark: MOOSE vs c++11-STL" << endl;
    //generate_sample( );
    benchmark_moose_tnmm_boost_vs_stl( 2, 10 );
    return 0;
}
