/*******************************************************************
 * File:            NormalRng.cpp
 * Description:      This is the MOOSE front end for class Normal,
 *                   which generates normally distributed random
 *                   doubles.
 * Author:          Subhasis Ray
 * E-mail:          ray.subhasis@gmail.com
 * Created:         2007-11-03 22:07:04
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

#ifndef _NORMALRNG_CPP
#define _NORMALRNG_CPP
#include "NormalRng.h"
#include "Normal.h"
#include "basecode/moose.h"
extern const Cinfo* initRandGeneratorCinfo();

const Cinfo* initNormalRngCinfo()
{
    static Finfo* normalRngFinfos[] =
        {
            new ValueFinfo("mean", ValueFtype1<double>::global(),
                           GFCAST( &NormalRng::getMean),
                           RFCAST( &NormalRng::setMean)),
            new ValueFinfo("variance", ValueFtype1<double>::global(),
                           GFCAST( &NormalRng::getVariance),
                           RFCAST( &NormalRng::setVariance)),
            new ValueFinfo("method", ValueFtype1<int>::global(),
                           GFCAST( &NormalRng::getMethod),
                           RFCAST( &NormalRng::setMethod)),
            
        };
    
	static string doc[] =
	{
		"Name", "NormalRng",
		"Author", "Subhasis Ray",
		"Description", "Normally distributed random number generator.",
	};    
	static Cinfo normalRngCinfo(
                                doc,
				sizeof( doc ) / sizeof( string ),
                                initRandGeneratorCinfo(),
                                normalRngFinfos,
                                sizeof(normalRngFinfos)/sizeof(Finfo*),
                                ValueFtype1<NormalRng>::global()
                                );
    return &normalRngCinfo;
}

    
static const Cinfo* normalRngCinfo = initNormalRngCinfo();

/**
   Set the mean of the internal generator object.   
 */
void NormalRng::setMean(const Conn* c, double mean)
{
    NormalRng* generator = static_cast < NormalRng* >(c->data());
    static_cast < Normal* > (generator->rng_)->setMean(mean);          
}

/**
   Since normal distribution is defined in terms of mean and variance, we
   want to store them in order to create the internal generator object.   
 */
void NormalRng::setVariance(const Conn* c, double variance)
{
    if ( variance < 0 )
    {
            cerr << "ERROR: variance cannot be negative." << endl;
            return;
    }
        
    NormalRng* generator = static_cast < NormalRng* >(c->data());
    static_cast < Normal* > (generator->rng_)->setVariance(variance);
}
/**
   Returns the algorithm used.
   0 for alias method.
   1 for BoxMueller method.
 */
NormalGenerator NormalRng::getMethod(Eref e)
{
    NormalRng* generator = static_cast <NormalRng*> (e.data());
    return static_cast<Normal*> (generator->rng_)->getMethod();
}
/**
   Set the algorithm to be used.
   1 for BoxMueller method.
   Anything else for alias method.
 */
void NormalRng::setMethod(const Conn* c, NormalGenerator method)
{
    NormalRng* generator = static_cast <NormalRng*> ( c->data());
    
    if ( generator->rng_)
    {
        cout << "Warning: Changing method after generator object has been created. Current method: " << static_cast <Normal*> (generator->rng_)->getMethod() << ". New method: " << method << endl;
    }
    static_cast <Normal*> (generator->rng_)->setMethod(method);
}

void NormalRng::innerReinitFunc(const Conn* c, ProcInfo info)
{
    // do nothing
}

/**
   By default the method used for normal distribution is alias method
   by Ahrens and Dieter. In order to use some method other than the
   default Alias method, one should call setMethod with a proper
   method index before calling reset ( reinit ). Since different
   methods create different random sequences, the combined sequence
   may not have the intended distribution. By default mean and
   variance are set to 0.0 and 1.0 respectively.
 */
NormalRng::NormalRng():RandGenerator()
{
    rng_ = new Normal();
}
#endif
