/*******************************************************************
 * File:            ExponentialRng.h
 * Description:
 * Author:          Subhasis Ray
 * E-mail:          ray.subhasis@gmail.com
 * Created:         2007-11-08 11:27:50
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

#ifndef _EXPONENTIALRNG_H
#define _EXPONENTIALRNG_H

#include "../basecode/header.h"
#include "randnum.h"
#include <random>

/**
   This is MOOSE wrapper for Exponentially distributed random number generator class, Exponential.
   The default
 */
class ExponentialRng
{
  public:

    ExponentialRng();

    ExponentialRng& operator=(const ExponentialRng&);

    double getMean() const;
    void setMean(double mean);

    void setSeed( int seed );
    int getSeed( ) const;

    virtual void vReinit( const Eref& e, ProcPtr p);
    static const Cinfo* initCinfo();

  private:
    double mean_;
    bool isMeanSet_;
    double seed_;

    std::mt19937 rng_;
    std::random_device rd_;

};


#endif
