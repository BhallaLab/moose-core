/*******************************************************************
 * File:            RandGenerator.h
 * Description:      
 * Author:          Subhasis Ray
 * E-mail:          ray.subhasis@gmail.com
 * Created:         2007-11-07 16:25:08
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

#ifndef _RANDGENERATOR_H
#define _RANDGENERATOR_H

#include "randnum.h"
#include "basecode/header.h"
#include "basecode/moose.h"
#include "Probability.h"

class RandGenerator
{
    
  public:
    RandGenerator();
    virtual ~RandGenerator();
    static double getMean(Eref e);    
    static double getVariance(Eref e);
    static double getSample(Eref e);
    static void processFunc( const Conn* c, ProcInfo info);
    static void reinitFunc( const Conn* c, ProcInfo info);
    virtual void innerReinitFunc( const Conn* c, ProcInfo info);    
  protected:
    Probability* rng_;    
};

    
#endif
