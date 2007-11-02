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
#include <cmath>
#include <iostream>

using namespace std;
static const unsigned int FAK_LEN = 1024;       // length of factorial table
static const double SHAT1 = 2.943035529371538573;    // 8/e
static const double SHAT2 = 0.8989161620588987408;   // 3-sqrt(12/e)
static const double EPSILON = getMachineEpsilon();

/***********************************************************************
Log factorial function - author Agner Fog. published under GPL
***********************************************************************/
double lnFac(unsigned long n) {
   // log factorial function. gives natural logarithm of n!

   // define constants
   static const double                 // coefficients in Stirling approximation     
      C0 =  0.918938533204672722,      // ln(sqrt(2*pi))
      C1 =  1./12., 
      C3 = -1./360.;
   // C5 =  1./1260.,                  // use r^5 term if FAK_LEN < 50
   // C7 = -1./1680.;                  // use r^7 term if FAK_LEN < 20
   // static variables
   static double fac_table[FAK_LEN];   // table of ln(n!):
   static int initialized = 0;         // remember if fac_table has been initialized

   if (n < FAK_LEN) {
      if (n <= 1)
      {
          if (n < 0)
          {
              cerr << "Parameter negative in LnFac function" << endl;  
          }          
         return 0;
      }
      if (!initialized) {              // first time. Must initialize table
         // make table of ln(n!)
         double sum = fac_table[0] = 0.;
         for (unsigned int i=1; i<FAK_LEN; i++) {
            sum += log(double(i));
            fac_table[i] = sum;
         }
         initialized = 1;
      }
      return fac_table[n];
   }
   // not found in table. use Stirling approximation
   double  n1, r;
   n1 = n;  r  = 1. / n1;
   return (n1 + 0.5)*log(n1) - n1 + C0 + r*(C1 + r*r*C3);
}


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
        if ( mean_ < 1.e-6 )
        {
            return poissonLow();
        } else
        {            
            return poissonInverse();
        }
    }
    
    return ratioOfUniforms();    
}
// Adapted from stocastic library by Agner Fog. http://www.agner.org/random/ which is published under GPL
double Poisson::poissonLow() const
{
    if (mean_ <= EPSILON)
    {
        return 0;
    }
    
    double d = sqrt(mean_);            
            
    if ( mtrand() >= d )
    {
        return 0;
    }
    double r = mtrand()*d;
    if ( r > mean_*(1.0-mean_))
    {
        return 1.0;
    }
    return 2.0;
}
// Adapted from stocastic library by Agner Fog. http://www.agner.org/random/ which is published under GPL
double Poisson::poissonInverse() const
{
    const int bound = 130;
    static double poissonF0 = exp(-mean_);
    double uniform;
    double result = 0.0;
    double fnValue;
    
    while (true)
    {
        uniform = mtrand();
        result = 0;
        fnValue = poissonF0;
        do 
        {
            uniform -= fnValue;
            if (uniform <= 0)
            {
                return result;
            }
            result++;
            fnValue *= mean_;
            uniform *= result;
        }
        while (result <= bound);
    }
    return result;
}
// Adapted from stocastic library by Agner Fog. http://www.agner.org/random/ which is published under GPL
double Poisson::ratioOfUniforms() const
{
    double uniform;
    double logarithm;
    double sample;
    unsigned long intSample;

    static double hatCentre = mean_ + 0.5;
    static unsigned long mode = (unsigned long)mean_;
    static double logMean = log(mean_);    
    static double poissonF0 = mode*logMean - lnFac(mode);
    static double hatWidth = sqrt(SHAT1 * (mean_+0.5)) + SHAT2;
    static unsigned long bound = (unsigned long)(hatCentre + 6.0 * hatWidth);
    while(true)
    {
        uniform = mtrand();
        if (abs(uniform) < EPSILON ) // avoid division by zero
        {
            continue;
        }
        sample = hatCentre + hatWidth*(mtrand() - 0.5)/uniform;
        if ( sample < 0 || sample >= bound )
        {
            continue;
        }
        intSample = (unsigned long)(sample);
        logarithm = intSample * logMean - lnFac(intSample) - poissonF0;
        if (logarithm >= uniform*(4.0-uniform) - 3.0)
        {
            break;
        }
        if ( uniform*(uniform - logarithm) > 1.0 )
        {
            continue;
        }
        if ( 2.0 * log(uniform) <= logarithm )
        {
            break;
        }
    }
    return intSample;
    
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
