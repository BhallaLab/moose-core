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
            new ValueFinfo("mean", ValueFtype1<double>::global(),
                           GFCAST( &PoissonRng::getMean),
                           RFCAST( &PoissonRng::setMean)),
        };
    static string doc[] =
	{
		"Name", "PoissonRng",
		"Author", "Subhasis Ray",
		"Description", "Poisson distributed random number generator.",
	};

    static Cinfo poissonRngCinfo(
                                doc,
				sizeof( doc ) / sizeof( string ),
                                initRandGeneratorCinfo(),
                                poissonRngFinfos,
                                sizeof(poissonRngFinfos)/sizeof(Finfo*),
                                ValueFtype1<PoissonRng>::global()
                                );
    return &poissonRngCinfo;
}

    
static const Cinfo* poissonRngCinfo = initPoissonRngCinfo();
PoissonRng::PoissonRng()
{
    //do nothing. should not try to get mean 
}

/**
   Sets the mean. Since poisson distribution is defined in terms of
   the rate parameter or the mean, it is mandatory to set this before
   using the generator.
*/
void PoissonRng::setMean(const Conn* c, double mean)
{
    PoissonRng* generator = static_cast<PoissonRng*>(c->data());
    if ( !generator->rng_ )
    {
        generator->rng_ = new Poisson(mean);
    }
    else
    {
        static_cast<Poisson*>(generator->rng_)->setMean(mean);
    }    
}
/**
   reports error in case the parameter mean has not been set.
*/
void PoissonRng::innerReinitFunc(const Conn* c, ProcInfo info)
{
    PoissonRng* generator = static_cast < PoissonRng* >(c->data());
    if (! generator->rng_ )
    {
        cerr << "ERROR: PoissonRng::innerReinitFunc - mean must be set before using the Poisson distribution generator." << endl;                
    }
}

    
#endif
