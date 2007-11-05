/*******************************************************************
 * File:            Poisson.h
 * Description:      
 * Author:          Subhasis Ray
 * E-mail:          ray.subhasis@gmail.com
 * Created:         2007-11-02 09:43:47
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

#ifndef _POISSON_H
#define _POISSON_H
#include "Probability.h"
class Poisson:public Probability
{
  public:
    Poisson(double mean);
    
    double getMean() const;
    double getVariance() const;
    double getNextSample() const;
  private:
    double mean_;
    double poissonSmall() const;
    double poissonLarge() const;
};

    
#endif
