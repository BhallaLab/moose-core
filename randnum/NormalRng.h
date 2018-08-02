/*******************************************************************
 * File:            NormalRng.h
 * Description:
 * Author:          Subhasis Ray, Dilawar Singh
 * E-mail:          ray.subhasis@gmail.com
 * Created:         2007-11-05 10:19:18
 * LOGS:
 *  Thursday 02 August 2018 05:03:10 PM IST
 *      - Uses <random> or boost library to genreate the distributions.
**********************************************************************/

#ifndef _NORMALRNG_H
#define _NORMALRNG_H

#include "randnum.h"
#include "RNG.h"
#include "../basecode/header.h"

/**
   This is MOOSE wrapper for normally distributed random number generator class, Normal.
   The default
 */
class NormalRng
{
  public:
    NormalRng();
    ~NormalRng() { ; }

    NormalRng& operator=( const NormalRng& );

    void setSeed(unsigned long seed);
    unsigned long getSeed( void ) const;

    void setMean(double mean);
    double getMean( void ) const;

    void setVariance(double variance);
    double getVariance( ) const;

    double getNextSample( );

    void reinitSeed( void );
    void reinit( const Eref& e, ProcPtr p);


    static const Cinfo * initCinfo();

  private:

    double mean_;
    double variance_;

    unsigned long seed_;

    moose::MOOSE_RANDOM_DEVICE rd_;
    moose::MOOSE_RNG_DEFAULT_ENGINE rng_;
    moose::MOOSE_NORMAL_DISTRIBUTION dist_;

};

#endif
