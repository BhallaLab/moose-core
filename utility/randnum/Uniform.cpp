/*******************************************************************
 * File:            Uniform.cpp
 * Description:      
 * Author:          Subhasis Ray
 * E-mail:          ray.subhasis@gmail.com
 * Created:         2008-02-21 17:12:55
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

#ifndef _UNIFORM_CPP
#define _UNIFORM_CPP
#include "Uniform.h"
#include "randnum.h"
#include <iostream>
using namespace std;

Uniform::Uniform()
{
    min_ = 0.0;
    max_ = 1.0;    
}
Uniform::Uniform(double min, double max)
{
    if ( min >= max )
    {
        cerr << "ERROR: specified lowerbound is greater than upper bound." << endl;
        min_ = 0.0;
        max_ = 1.0;
        return;
    }
    
    min_ = min;
    max_ = max;
}
double Uniform::getMean() const
{
    return (max_ - min_)/2.0;
}
double Uniform::getVariance()const
{
    return (max_-min_)*(max_ - min_)/12.0;
}
double Uniform::getMin()
{
    return min_;
}
double Uniform::getMax()
{
    return max_;
}
void Uniform::setMin(double min)
{
    min_ = min;
}
void Uniform::setMax(double max)
{
    max_ = max;
}
double Uniform::getNextSample()
{
    return mtrand()*(max_-min_)+min_;
}


#endif
