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
//     static Finfo* processShared[] = 
//         {
//             new DestFinfo("process", Ftype1<ProcInfo>::global(),
//                           RFCAST( &NormalRng::processFunc)),
//             new DestFinfo("reinit", Ftype1<ProcInfo>::global(),
//                           RFCAST( &NormalRng::reinitFunc)),
//         };
//     static Finfo* process = new SharedFinfo( "process", processShared,
//                                              sizeof(processShared)/sizeof(Finfo*));
    static Finfo* normalRngFinfos[] =
        {
            new ValueFinfo("sample", ValueFtype1<double>::global(),
                           GFCAST( &NormalRng::getSample),
                           RFCAST(&dummyFunc)),
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
    
    static Cinfo normalRngCinfo("NormalRng",
                                "Subhasis Ray",
                                "Normally distributed random number generator.",
                                initRandGeneratorCinfo(),
                                normalRngFinfos,
                                sizeof(normalRngFinfos)/sizeof(Finfo*),
                                ValueFtype1<NormalRng>::global()
                                );
    return &normalRngCinfo;
}

    
static const Cinfo* normalRngCinfo = initNormalRngCinfo();


double NormalRng::getSample( const Element* e)
{
    return static_cast<NormalRng*>( e->data())->normalRng_->getNextSample();
}

double NormalRng::getMean( const Element *e)
{
    return static_cast<NormalRng*>(e->data())->mean_;
}

void NormalRng::setMean(const Conn& c, double mean)
{
    NormalRng* generator = static_cast < NormalRng* >(c.data());
    
    generator->mean_ = mean;
    generator->isMeanSet_ = true;
    if ( generator->isVarianceSet_&& (!generator->normalRng_))
    {
        generator->normalRng_ = new Normal(generator->method_, generator->mean_, generator->variance_);
    }
    else if ( generator->normalRng_)
    {
        generator->mean_ = generator->normalRng_->getMean();        
    }    
}

double NormalRng::getVariance(const Element* e)
{
    return static_cast<NormalRng*>(e->data())->variance_;
}

void NormalRng::setVariance(const Conn& c, double variance)
{
    if ( variance < 0 )
    {
            cerr << "ERROR: variance cannot be negative." << endl;
            return;
    }
        
    NormalRng* generator = static_cast < NormalRng* >(c.data());
    generator->isVarianceSet_ = true;
    if ( generator->isMeanSet_ && (!generator->normalRng_))
    {
        generator->normalRng_ = new Normal(generator->method_, generator->mean_, variance);        
        generator->variance_ = variance;
    }
}
int NormalRng::getMethod(const Element* e)
{
    return static_cast <NormalRng*> (e->data())->method_;
}

void NormalRng::setMethod(const Conn& c, int method)
{
    NormalRng* generator = static_cast <NormalRng*> ( c.data());
    
    if (! generator->normalRng_)
    {
        switch ( method )
        {
            case 0:
                generator->method_ = ALIAS;
                break;
            default:
                generator->method_ = BOX_MUELLER;
                break;
        }
    }
    else 
    {
        cerr << "Warning: cannot change method after generator object has been created. Method in use: " << static_cast <NormalRng*> ( c.data())->method_ << endl;
    }
}

// void NormalRng::processFunc( const Conn& c, ProcInfo info )
// {
//     send1<double>(c.targetElement(), outputSlot, getSample(c.targetElement()));    
// }

void NormalRng::reinitFunc(const Conn& c, ProcInfo info)
{
    NormalRng* generator = static_cast < NormalRng* >(c.data());
    if (! generator->normalRng_ )
    {
        if (generator->isMeanSet_ && generator->isVarianceSet_)
        {
            generator->normalRng_ = new Normal(generator->method_, generator->mean_, generator->variance_);
        } else 
        {
            cerr << "WARNING: creating default standard normal distribution generator." << endl;            
            generator->normalRng_ = new Normal();
        }        
    }
}

/**
   By default the method used for normal distribution is alias method
   by Ahrens and Dieter.  Mean and variance values are not presumed -
   the user must explicitly specify these before trying to obtain a
   sample.  The actual generator object is created only when both mean
   and variance are set. To avoid checking for these settings each
   time inside the process() call, we have booleans isMeanSet_ and
   isVarianceSet_ to keep track of the mean and variance status. As
   soon as both are set we instantiate Normal class with the mean_,
   variance_ and method_ as parameters. In order to use some method
   other than the default Alias method, one must call setMethod with a
   proper method index before setting mean or variance.
 */
NormalRng::NormalRng()
{
    isMeanSet_ = false;
    isVarianceSet_ = false;
    normalRng_ = 0;
    method_ = ALIAS;    
}
NormalRng::~NormalRng()
{
    delete normalRng_;    
}

#endif
