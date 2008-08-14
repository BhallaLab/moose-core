/*******************************************************************
 * File:            UniformRng.cpp
 * Description:      
 * Author:          Subhasis Ray
 * E-mail:          ray.subhasis@gmail.com
 * Created:         2008-02-01 11:30:20
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

#ifndef _UNIFORMRNG_CPP
#define _UNIFORMRNG_CPP
#include "randnum.h"
#include "UniformRng.h"
#include "basecode/moose.h"
extern const Cinfo* initRandGeneratorCinfo();

const Cinfo* initUniformRngCinfo()
{
    static Finfo* uniformRngFinfos[] =
        {
            
            new ValueFinfo("mean", ValueFtype1<double>::global(),
                           GFCAST( &UniformRng::getMean),
                           RFCAST( &dummyFunc)),
            new ValueFinfo("variance", ValueFtype1<double>::global(),
                           GFCAST( &UniformRng::getVariance),
                           RFCAST( &dummyFunc)),
            /// The lower bound on the numbers generated 
            new ValueFinfo("min", ValueFtype1 <double>::global(),
                           GFCAST( &UniformRng::getMin),
                           RFCAST( &UniformRng::setMin)),
            /// The upper bound on the numbers generated
            new ValueFinfo("max", ValueFtype1 <double>::global(),
                           GFCAST( &UniformRng::getMax),
                           RFCAST( &UniformRng::setMax)),
                           
        };
    
    
    static Cinfo uniformRngCinfo("UniformRng",
                                "Subhasis Ray",
                                "Uniformly distributed random number generator.",
                                initRandGeneratorCinfo(),
                                uniformRngFinfos,
                                sizeof(uniformRngFinfos)/sizeof(Finfo*),
                                ValueFtype1<UniformRng>::global()
                                );
    return &uniformRngCinfo;
}

static const Cinfo* uniformRngCinfo = initUniformRngCinfo();

double UniformRng::getMean(const Element * e)
{
    UniformRng* rng = static_cast<UniformRng*> (e->data());
    if ( rng )
    {
        return (rng->min_+ rng->max_)/2.0;
    }
    return 0.0;    
}

    
double UniformRng::getVariance(const Element* e)
{
    UniformRng* rng = static_cast<UniformRng*> (e->data());
    if ( rng )
    {
        return (rng->max_ - rng->min_)*(rng->max_ - rng->min_)/12.0;
    }
    return -1.0; // error    
}


double UniformRng::getMin(const Element * e)
{
    return static_cast<UniformRng*> (e->data())->min_;
}

double UniformRng::getMax(const Element * e)
{
    return static_cast<UniformRng*> (e->data())->max_;
}

void UniformRng::setMin(const Conn& c, double min)
{
    UniformRng* obj = static_cast <UniformRng*> (c.data());
    if (obj)
    {
        obj->min_ = min;
    }
    else
    {
        cerr << "UniformRng::setMin() - connection target is NULL." << endl;
    }
}
void UniformRng::setMax(const Conn& c, double max)
{
    UniformRng* obj = static_cast <UniformRng*> (c.data());
    if (obj)
    {
        obj->max_ = max;
    }
    else
    {
        cerr << "UniformRng::setMax() - connection target is NULL." << endl;
    }
}

double UniformRng::getSample(const Element* e)
{
    UniformRng* rng = static_cast<UniformRng*> (e->data());
    if ( rng)
    {
        return mtrand()*( rng->max_ - rng->min_ ) + rng->min_;
    }
    else
    {
        cerr << "UniformRng::getSample() - Element is NULL." << endl;
        return 0.0; // Error condition
    }    
}

UniformRng::UniformRng():RandGenerator()
{
    min_ = 0.0;
    max_ = 1.0;
}

UniformRng::~UniformRng()
{
}

    
#endif
