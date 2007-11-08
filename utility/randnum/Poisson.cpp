/*******************************************************************
 * File:            Poisson.cpp
 * Description:      
 * Author:          Subhasis Ray
 * E-mail:          ray.subhasis@gmail.com
 * Created:         2007-11-01 16:09:38
 ********************************************************************/
/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2005 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU General Public License version 2
** See the file COPYING.LIB for the full notice.
**********************************************************************/
/******************************************************************
 * Some of the functions have been copied/adapted from stochastic
 * library package by Agner Fog, published under GPL.
 * Refer to http://www.agner.org/random/ for more information.
 ******************************************************************/
#ifndef _POISSON_CPP
#define _POISSON_CPP
#include "Poisson.h"
#include "randnum.h"
#include "NumUtil.h"
#include "Gamma.h"
#include "Binomial.h"
#include <cmath>
#include <iostream>

using namespace std;




Poisson::Poisson(double mean):mean_(mean)
{
}

double Poisson::getMean() const
{
    return mean_;
}

double Poisson::getVariance() const
{
    return mean_;
}

double Poisson::getNextSample() const
{
    if (mean_ < 17)
    {
        return poissonSmall();
     
    }
    
    return poissonLarge();    
}

/**
   Poisson distributed random number generator when mean is small.
   See: TAOCP by Knuth, Volume 2, Section 3.4.1
 */
double Poisson::poissonSmall() const
{
    static double  limit = exp(-mean_);
    
    double product = 1.0;
    
    int i = 0;
    while ( product > limit )
    {
        product *= mtrand();
        ++i;        
    }
    return i; 
}

/**
   Poisson distributed random number generator when mean is large.
   See: TAOCP by Knuth, Volume 2, Section 3.4.1
 */
double Poisson::poissonLarge() const
{

    // generate X with the gamma distribution of order floor(alpha*mu)
    // where alpha is a suitable constant
    static int m_value = (int)floor(0.875*mean_); // alpha = 7/8 = 0.875 is a good value according to Ahrens and Dieter
    static Gamma gammaGen(m_value, 1.0);
    double n_value;
    
    double x_value = gammaGen.getNextSample();    
    
    if ( x_value < mean_ )
    {
        Poisson poissonGen(mean_ - x_value);        
        n_value = m_value+ poissonGen.getNextSample();
    }
    else
    {
        Binomial binomialGen(m_value-1,mean_/x_value);
        n_value = binomialGen.getNextSample();        
    }
    
    return n_value;    
}

#if 0 // test main

#include <vector>
#include <algorithm>
#include <cstdlib>
int main(int argc, char **argv)
{
    double mean = 4;
    
    if (argc > 1)
        mean = atof(argv[1]);
    
        
    double sum = 0.0;
    double sd = 0.0;    
    vector <double> samples;
    Poisson poisson(mean);
    int MAX_SAMPLE = 100000;
    unsigned index;
    
    
    cout << "epsilon = " << getMachineEpsilon() << endl;
    for ( int i = 0; i < MAX_SAMPLE; ++i )
    {
        double p = poisson.getNextSample();//aliasMethod();
        samples.push_back(p);
                
        sum += p;
        sd += (p - mean)*(p - mean);        
    }
    mean = sum/MAX_SAMPLE;
    sd = sqrt(sd/MAX_SAMPLE);
    cout << "mean = " << mean << " sd = " << sd << endl;
    sort(samples.begin(), samples.end());

    unsigned i = 0;
    unsigned start = 0;
    index = 0;
    
    while ( i < samples.size() )
    {
        int count = 0;
        
        while( ( i < samples.size() ) && (samples[i] == samples[start] ))
        {
            ++count;
            ++i;
        }
        while( index < samples[start])
        {
            cout << index++ << " " << 0 << endl;
        }
        
        cout << index++ << " " << count << endl;        
        start = i;     
    }
    return 0;
}
#endif // test main
#endif
