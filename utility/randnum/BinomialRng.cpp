/*******************************************************************
 * File:            BinomialRng.cpp
 * Description:      
 * Author:          Subhasis Ray
 * E-mail:          ray.subhasis@gmail.com
 * Created:         2007-11-08 10:58:01
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

#ifndef _BINOMIALRNG_CPP
#define _BINOMIALRNG_CPP
#include "BinomialRng.h"
#include "Binomial.h"

extern const Cinfo* initRandGeneratorCinfo();

const Cinfo* initBinomialRngCinfo()
{
    static Finfo* binomialRngFinfos[] =
        {
            new ValueFinfo("n", ValueFtype1<int>::global(),
                           GFCAST( &BinomialRng::getN),
                           RFCAST(&BinomialRng::setN)),
            new ValueFinfo("p", ValueFtype1<double>::global(),
                           GFCAST( &BinomialRng::getP),
                           RFCAST(&BinomialRng::setP)),
        };
    
    static Cinfo binomialRngCinfo("BinomialRng",
                                  "Subhasis Ray",
                                  "Binomially distributed random number generator.",
                                  initRandGeneratorCinfo(),
                                  binomialRngFinfos,
                                  sizeof(binomialRngFinfos)/sizeof(Finfo*),
                                  ValueFtype1<BinomialRng>::global()
        );
    return &binomialRngCinfo;
}

    
static const Cinfo* binomialRngCinfo = initBinomialRngCinfo();

BinomialRng::BinomialRng()
{
    isNSet_ = false;
    isPSet_ = false;
    n_ = 0;
    p_ = 0;    
}

/**
   Set parameter n ( number of trials for a two-outcome experiment).
   This must be set before the actual generator is instantiated.
 */
void BinomialRng::setN(const Conn& c, int n)
{    
    if ( n <= 0 )
    {
        cerr << "ERROR: BinomialRng::setN - n must be a positive integer." << endl;
        return;
    }
    BinomialRng* gen = static_cast<BinomialRng*>(c.data());
    gen->isNSet_ = true;
    gen->n_ = n;
    if ( gen->isNSet_ && gen->isPSet_)
    {
        if ( !gen->rng_ )
        {
            gen->rng_ = new Binomial((unsigned long)gen->n_,gen->p_);
        }
        else
        {
            cerr << "WARNING: cannot modify parameter after initialization." << endl;
            gen->n_ = static_cast<Binomial*>(gen->rng_)->getN();            
        }
    }    
}
/**
   Returns parameter n.
 */
int BinomialRng::getN(const Element* e)
{
    return (int)(static_cast <BinomialRng*>(e->data())->n_);
}
/**
   Set parameter p ( the probability of the outcome of interest ).
   This must be set before the actual generator is instantiated.
 */
void BinomialRng::setP(const Conn& c, double p)
{
 
    if ( p < 0 || p > 1)
    {
        cerr << "ERROR: BinomialRng::setP - p must be in (0,1) range." << endl;
        return;
    }
    BinomialRng* gen = static_cast<BinomialRng*>(c.data());
    gen->isPSet_ = true;
    gen->p_ = p;
    
    if ( gen->isNSet_ && gen->isPSet_)
    {
        if ( !gen->rng_ )
        {
            gen->rng_ = new Binomial((unsigned long)(gen->n_),gen->p_);
        }
        else
        {
            cerr << "WARNING: BinomialRng::setP - cannot modify parameter after initialization." << endl;
            gen->p_ = static_cast<Binomial*>(gen->rng_)->getP();            
        }        
    }
}

/**
   returns parameter p.
*/
double BinomialRng::getP(const Element* e)
{
    return static_cast <BinomialRng*>(e->data())->p_;
}

/**
   reports error if one or more of the parameters are not set.
*/
void BinomialRng::reinitFunc( const Conn& c, ProcInfo info)
{
    BinomialRng* gen = static_cast<BinomialRng*>(c.data());

    if ( gen->isNSet_ )
    {
        if ( gen->isPSet_ )
        {
            if ( !gen->rng_ )
                gen->rng_ = new Binomial((unsigned long)(gen->n_),gen->p_);
        }
        else 
        {
            cerr << "ERROR: BinomialRng::reinit - first set value of p." << endl;
        }
    }
    else 
    {
        cerr << "ERROR: BinomialRng::reinit - first set value of n." << endl;
    }
}



#endif
