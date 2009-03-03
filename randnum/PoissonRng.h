/*******************************************************************
 * File:            PoissonRng.h
 * Description:      
 * Author:          Subhasis Ray
 * E-mail:          ray.subhasis@gmail.com
 * Created:         2007-11-07 16:22:35
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

#ifndef _POISSONRNG_H
#define _POISSONRNG_H
#include "Poisson.h"
#include "RandGenerator.h"
class PoissonRng:public RandGenerator
{
  public:
    PoissonRng();
    virtual ~PoissonRng() { ; }
    static void setMean(const Conn* c, double mean);
    virtual void innerReinitFunc(const Conn* c, ProcInfo p);
};


#endif
