/*******************************************************************
 * File:            ExponentialRng.cpp
 * Description:
 * Author:          Subhasis Ray
 * E-mail:          ray.subhasis@gmail.com
 * Created:         2007-11-08 11:33:45
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

#ifndef _EXPONENTIALRNG_CPP
#define _EXPONENTIALRNG_CPP

#include "ExponentialRng.h"

const Cinfo* ExponentialRng::initCinfo()
{
    static ValueFinfo< ExponentialRng, double > rate(
        "rate",
        "Rate parameter (ƛ) of the exponential distribution.",
        &ExponentialRng::setRate,
        &ExponentialRng::getRate
        );

    static ValueFinfo< ExponentialRng, int > seed(
        "seed",
        "Set the seed for RNG. If not set, it uses value passed to moose.seed( ) function.",
        &ExponentialRng::setSeed,
        &ExponentialRng::getSeed
        );

    static Finfo* exponentialRngFinfos[] = {
        &rate,
        &seed,
    };

    static string doc[] = {
        "Name", "ExponentialRng",
        "Author", "Subhasis Ray, Dilawar Singh",
        "Description", "Exponentially distributed random number generator.\n"
        "Exponential distribution with mean k is defined by the probability"
        " density function p(x; k) = k * exp(-k * x) if x >= 0, else 0."
    };

    static Dinfo< ExponentialRng > dinfo;
    static Cinfo exponentialRngCinfo(
        "ExponentialRng",
        Neutral::initCinfo(),
        exponentialRngFinfos,
        sizeof(exponentialRngFinfos)/sizeof(Finfo*),
        &dinfo,
        doc,
        sizeof( doc ) / sizeof( string ));
    return &exponentialRngCinfo;
}


static const Cinfo* exponentialRngCinfo = ExponentialRng::initCinfo();

ExponentialRng::ExponentialRng()
{
    rate_ = 0;
}

ExponentialRng& ExponentialRng::operator=(const ExponentialRng& r)
{
    seed_ = r.seed_;
    rate_ = r.rate_;
    return *this;
}

int ExponentialRng::getSeed( void ) const
{
    return seed_;
}

void ExponentialRng::setSeed( int seed )
{
    seed_ = seed;
}


/**
   Replaces the same method in base class.  Returns the mean as
   stored in this object independent of the actual generator object.
 */
double ExponentialRng::getRate() const
{
    return rate_;
}
/**
   Sets the mean. Since exponential distribution is defined in terms
   of this parameter, it is stored locally independent of the
   instantiation of the internal generator object.
*/
void ExponentialRng::setRate(double mean)
{
    rate_ = mean;
}

void ExponentialRng::reinitSeed()
{
    if( seed_ >= 0 )
    {
        rng_.seed( seed_ );
        return;
    }

    if( moose::getGlobalSeed() >= 0 )
    {
        rng_.seed( moose::getGlobalSeed() );
        return;
    }

    rng_.seed( rd_() );
}

/**
   Reports error in case the parameter mean has not been set.
 */
void ExponentialRng::reinit(const Eref& e, ProcPtr p)
{
    reinitSeed();
    // Reinit <random>
}

#endif
