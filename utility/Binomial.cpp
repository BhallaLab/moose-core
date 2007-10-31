/*******************************************************************
 * File:            Binomial.cpp
 * Description:      
 * Author:          Subhasis Ray
 * E-mail:          ray.subhasis@gmail.com
 * Created:         2007-10-28 13:44:46
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

#ifndef _BINOMIAL_CPP
#define _BINOMIAL_CPP
#include <cmath>
#include "randnum.h"
#include "Binomial.h"

Binomial::Binomial(unsigned long n, double p, long seed):n_(n),p_(p)
{    
}

unsigned long Binomial::getN()
{
    return n_;    
}

double Binomial::getP()
{
    return p_;
}

double Binomial::getMean()
{
    return n_*p_;    
}

double Binomial::getVariance()
{
    return sqrt(n_*p_*(1.0-p_));
}
/**
   returns the next random number in this distribution as the ratio of
   the number of positive outcomes and the total number of trials.
*/
double Binomial::getNextSample()
{
    long double sample = 0;
    
    for ( unsigned int i = 0; i < n_; ++i)
    {
        double myRand = mtrand();
        if ( myRand < p_ )
        {
            sample+=1;
        }        
    }
    return sample/n_;    
}

#include <iostream>
#include <climits>
using namespace std;

/**
   TODO: what to do to automatically test the quality of the random
   number generation? We can check the mean and variance perhaps?
   We should also check the plot of the distribution manually.
 */
void testBinomial()
{

    int trialMin = 10;
    int trialMax = trialMin*1000;
    
    double tmp;

    
    for ( int i = trialMin; i < trialMax; i =(int)( i* 1.1) )
    {
        for ( double p = 0.1; p < 1.0; p += 0.05)
        {
            Binomial b(i, p);
            tmp = 0;
            for ( int j = 0; j < i; ++j )
            {
                tmp += b.getNextSample();            
            }
            cerr << "Diff( " << i << "," << p << ") "
                 << tmp - b.getMean()
                 << " [ " << tmp << " - " << b.getMean() <<" ]"
                 << endl;   
        }        
    }
}
#if 0 // test main
int main(void)
{
    testBinomial(); 
    return 0;    
}

#endif // test main


#endif
