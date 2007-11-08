/*******************************************************************
 * File:            PoissonRng.cpp
 * Description:      
 * Author:          Subhasis Ray
 * E-mail:          ray.subhasis@gmail.com
 * Created:         2007-11-08 09:53:32
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
#include "basecode/header.h"
#include "PoissonRng.h"
#include "basecode/moose.h"
extern const Cinfo* initRandGeneratorCinfo();

const Cinfo* initPoissonRngCinfo()
{
    static Finfo* poissonRngFinfos[] =
        {
            new ValueFinfo("sample", ValueFtype1<double>::global(),
                           GFCAST( &PoissonRng::getSample),
                           RFCAST(&dummyFunc)),
            new ValueFinfo("mean", ValueFtype1<double>::global(),
                           GFCAST( &PoissonRng::getMean),
                           RFCAST( &PoissonRng::setMean)),
        };
    
    static Cinfo poissonRngCinfo("PoissonRng",
                                "Subhasis Ray",
                                "Poissonly distributed random number generator.",
                                initRandGeneratorCinfo(),
                                poissonRngFinfos,
                                sizeof(poissonRngFinfos)/sizeof(Finfo*),
                                ValueFtype1<PoissonRng>::global()
                                );
    return &poissonRngCinfo;
}

    
static const Cinfo* poissonRngCinfo = initPoissonRngCinfo();

    
void PoissonRng::setMean(const Conn& c, double mean)
{
    PoissonRng* generator = static_cast<PoissonRng*>(c.data());
    if ( generator->rng_ )
    {
        delete generator->rng_;        
    }
    generator->rng_ = new Poisson(mean);
}
void PoissonRng::reinitFunc(const Conn& c, ProcInfo info)
{
    PoissonRng* generator = static_cast < PoissonRng* >(c.data());
    if (! generator->rng_ )
    {
        cerr << "ERROR: PoissonRng::reinitFunc - mean must be set before using the Poisson distribution generator." << endl;                
    } 
}

    
#endif
