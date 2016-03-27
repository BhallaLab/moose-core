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
#include "../randnum/randnum.h"
using namespace std;

int main(int argc, char *argv[])
{
    vector<double> randVec( 500000000 );
    cerr << "here we are" << endl;
    clock_t startT = clock();
    for( unsigned int i = 0; i < 500000000; ++i )
        randVec[i] = mtrand();
    cerr << "Time taken (MOOSE) " << clock() - startT << endl;

    // Using c++ library
    mt19937 rng( 0 );
    startT = clock();
    for( unsigned int i = 0; i < 500000000; ++i )
        randVec[i] = mtrand();
    cerr << "Time taken (STL) " << clock() - startT << endl;
    
    return 0;
}
