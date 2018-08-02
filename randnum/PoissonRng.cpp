/*******************************************************************
 * File:            PoissonRng.cpp
 * Description:
 * Author:          Subhasis Ray
 * E-mail:          ray.subhasis@gmail.com
 * Created:         2007-11-08 09:53:32
 * Updated:         2014-09-29
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

#ifndef _POISSONRNG_CPP
#define _POISSONRNG_CPP

#include "../basecode/header.h"

#include "RNG.h"
#include "PoissonRng.h"

const Cinfo* PoissonRng::initCinfo()
{
    static ValueFinfo< PoissonRng, double > mean(
        "mean",
        "Mean of the Poisson distribution.",
        &PoissonRng::setMean,
        &PoissonRng::getMean
        );

    static Finfo* poissonRngFinfos[] = {
        &mean,
    };

    static string doc[] = {
        "Name", "PoissonRng",
        "Author", "Subhasis Ray, Dilawar Singh",
        "Description", "Poisson distributed random number generator.",
    };

    static Dinfo< PoissonRng > dinfo;
    static Cinfo poissonRngCinfo(
        "PoissonRng",
        Neutral::initCinfo(),
        poissonRngFinfos,
        sizeof(poissonRngFinfos)/sizeof(Finfo*),
        &dinfo,
        doc,
        sizeof( doc ) / sizeof( string )
        );

    return &poissonRngCinfo;
}


static const Cinfo* poissonRngCinfo = PoissonRng::initCinfo();

PoissonRng::PoissonRng()
{
    //do nothing. should not try to get mean
}

PoissonRng& PoissonRng::operator=(const PoissonRng& rng)
{
    return *this;
}

/**
   Sets the mean. Since poisson distribution is defined in terms of
   the rate parameter or the mean, it is mandatory to set this before
   using the generator.
*/
void PoissonRng::setMean(double mean)
{
    mean_ = mean;
}

double PoissonRng::getMean( void ) const
{
    return mean_;
}

void PoissonRng::reinitSeed( void )
{

}

void PoissonRng::reinit(const Eref& e, ProcPtr p)
{
    reinitSeed();
    dist_ = moose::MOOSE_POISSON_DISTRIBUTION(mean_);
}


#endif
