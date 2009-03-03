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
extern const Cinfo* initRandGeneratorCinfo();

const Cinfo* initExponentialRngCinfo()
{
    static Finfo* exponentialRngFinfos[] =
        {
            new ValueFinfo("mean", ValueFtype1<double>::global(),
                           GFCAST( &ExponentialRng::getMean),
                           RFCAST( &ExponentialRng::setMean)),
            new ValueFinfo("method", ValueFtype1<int>::global(),
                            GFCAST(&ExponentialRng::getMethod),
                            RFCAST(&ExponentialRng::setMethod)),
        };
    
    static string doc[] =
	{
		"Name", "ExponentialRng",
		"Author", "Subhasis Ray",
		"Description", "Exponentially distributed random number generator.",
	};
    static Cinfo exponentialRngCinfo(
                                doc,
				sizeof( doc ) / sizeof( string ),
                                initRandGeneratorCinfo(),
                                exponentialRngFinfos,
                                sizeof(exponentialRngFinfos)/sizeof(Finfo*),
                                ValueFtype1<ExponentialRng>::global()
                                );
    return &exponentialRngCinfo;
}

    
static const Cinfo* exponentialRngCinfo = initExponentialRngCinfo();

ExponentialRng::ExponentialRng()
{
    mean_ = 0;    
    isMeanSet_ = false;
    method_ = RANDOM_MINIMIZATION;    
}
/**
   Replaces the same method in base class.  Returns the mean as
   stored in this object independent of the actual generator object.
 */
double ExponentialRng::getMean(Eref e)
{
    return static_cast<ExponentialRng*> (e.data())->mean_;    
}
/**
   Sets the mean. Since exponential distribution is defined in terms
   of this parameter, it is stored locally independent of the
   instantiation of the internal generator object.
*/
void ExponentialRng::setMean(const Conn* c, double mean)
{
    ExponentialRng* generator = static_cast<ExponentialRng*>(c->data());
    if ( !generator->rng_ )
    {
        generator->rng_ = new Exponential(mean);
        generator->isMeanSet_ = true;        
    }    
}
/**
   Reports error in case the parameter mean has not been set.
 */
void ExponentialRng::innerReinitFunc(const Conn* c, ProcInfo info)
{
    ExponentialRng* generator = static_cast < ExponentialRng* >(c->data());
    if (! generator->rng_ )
    {
        cerr << "ERROR: ExponentialRng::innerReinitFunc - mean must be set before using the Exponential distribution generator." << endl;                
    }
}

/**
   Returns the algorithm used for sample generation.
   0 for logarithmic method.
   1 for random minimization method.
 */
int ExponentialRng::getMethod(Eref e)
{
   return static_cast<ExponentialRng*>(e.data())->method_;    
}

/**
   Sets the algorithm used for sample generation.
   0 for logarithmic method.
   1 for random minimization method.
   Default is random minimization.
 */
void ExponentialRng::setMethod(const Conn* c, int method)
{
    ExponentialRng* generator = static_cast <ExponentialRng*> ( c->data());
    
    if (! generator->rng_)
    {
        switch ( method )
        {
            case 0:
                generator->method_ = LOGARITHMIC;
                break;
            default:
                generator->method_ = RANDOM_MINIMIZATION;                
                break;
        }
    }
    else 
    {
        cerr << "Warning: cannot change method after generator object has been created. Method in use: " << static_cast <ExponentialRng*> ( c->data())->method_ << endl;
    }
}


#endif
