/***
 *       Filename:  test_randnum.cpp
 *
 *    Description:  Test randnum support in moose.
 *
 *         Author:  Dilawar Singh <dilawars@ncbs.res.in>
 *   Organization:  NCBS Bangalore
 *        License:  GNU GPL3
 */

#include <iostream>
#include <vector>
#include <cmath>

#include "randnum.h"
#include "Exponential.h"

using namespace std;

int test_exponential(void)
{
    double mean = .25;
    double sum = 0.0;
    double sd = 0.0;
    vector <unsigned> classes;
    Exponential ex(mean);
    int MAX_SAMPLE = 100000;
    int MAX_CLASSES = 1000;


    for ( int i = 0; i < MAX_CLASSES; ++i )
    {
        classes.push_back(0);
    }

    for ( int i = 0; i < MAX_SAMPLE; ++i )
    {
        double p = ex.getNextSample();//aliasMethod();
        int index = (int)(p*MAX_CLASSES);
//        cout << index << " ] " << p << endl;

        if ( index < MAX_CLASSES){
            classes[index]++;
        }
        else
        {
            classes[MAX_CLASSES-1]++;
        }


        sum += p;
        sd += (p - mean)*(p - mean);
    }
    mean = sum/MAX_SAMPLE;
    sd = sqrt(sd/MAX_SAMPLE);
    cout << "mean = " << mean << " sd = " << sd << endl;
    for ( int i = 0; i < MAX_CLASSES; ++i )
    {
        cout << classes[i] << endl;
    }

    return 0;
}

int main(int argc, const char *argv[])
{
    return 0;
}
