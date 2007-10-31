/*******************************************************************
 * File:            Binomial.h
 * Description:      Implements binomial distribution
 * Author:          Subhasis Ray
 * E-mail:          ray.subhasis@gmail.com
 * Created:         2007-10-28 13:42:24
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

#ifndef _BINOMIAL_H
#define _BINOMIAL_H
#include "Probability.h"

class Binomial:Probability
{
  public:
    Binomial(){};
    Binomial( unsigned long n, double p);
    unsigned long getN();
    double getP();        
    double getMean();
    double getVariance();
    double getNextSample();
    
  private:
    unsigned long n_;
    double p_;
    
};

    
#endif
