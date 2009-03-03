/*******************************************************************
 * File:            GammaRng.h
 * Description:      
 * Author:          Subhasis Ray
 * E-mail:          ray.subhasis@gmail.com
 * Created:         2007-11-08 11:53:29
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

#ifndef _GAMMARNG_H
#define _GAMMARNG_H
#include "randnum.h"
#include "basecode/header.h"
#include "basecode/moose.h"
#include "RandGenerator.h"
#include "Gamma.h"

/**
   This is MOOSE wrapper for Gammaly distributed random number generator class, Gamma.
   The default
 */
class GammaRng: public RandGenerator
{
  public:
    GammaRng();
    virtual ~GammaRng() { ; }
    static double getAlpha(Eref e);
    static double getTheta(Eref e);
    static void setAlpha(const Conn* c, double alpha);
    static void setTheta(const Conn* c, double theta);
    
    virtual void innerReinitFunc( const Conn* c, ProcInfo info);

  private:
    double alpha_;
    double theta_;
    
    bool isAlphaSet_;
    bool isThetaSet_;    
};


#endif
