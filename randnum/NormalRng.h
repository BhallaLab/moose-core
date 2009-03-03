/*******************************************************************
 * File:            NormalRng.h
 * Description:      
 * Author:          Subhasis Ray
 * E-mail:          ray.subhasis@gmail.com
 * Created:         2007-11-05 10:19:18
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

#ifndef _NORMALRNG_H
#define _NORMALRNG_H
#include "randnum.h"
#include "basecode/header.h"
#include "basecode/moose.h"
#include "RandGenerator.h"
#include "Normal.h"

/**
   This is MOOSE wrapper for normally distributed random number generator class, Normal.
   The default
 */
class NormalRng: public RandGenerator
{
  public:
    NormalRng();
    virtual ~NormalRng() { ; }
    static void setMean(const Conn* c, double mean);
    static void setVariance(const Conn* c, double variance);
    static void setMethod(const Conn* c, NormalGenerator method);
    static NormalGenerator getMethod(Eref e);
    virtual void innerReinitFunc( const Conn* c, ProcInfo info);    
};

    
#endif
