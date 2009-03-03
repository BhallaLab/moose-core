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
#include <cmath>
extern const Cinfo* initRandGeneratorCinfo();

const Cinfo* initBinomialRngCinfo()
{
    static Finfo* binomialRngFinfos[] =
        {
            new ValueFinfo("n", ValueFtype1< double >::global(),
                           GFCAST( &BinomialRng::getN),
                           RFCAST(&BinomialRng::setN)),
            new ValueFinfo("p", ValueFtype1< double >::global(),
                           GFCAST( &BinomialRng::getP),
                           RFCAST(&BinomialRng::setP)),
            new DestFinfo("nDest", Ftype1 < double >::global(),
                          RFCAST(&BinomialRng::setN)),
            new DestFinfo("pDest", Ftype1 < double >::global(),
                          RFCAST(&BinomialRng::setP)),
        };
    
    static string doc[] =
	{
		"Name", "BinomialRng",
		"Author", "Subhasis Ray",
		"Description", "Binomially distributed random number generator.",
	};

    static Cinfo binomialRngCinfo(
                                  doc,
				  sizeof( doc ) / sizeof( string ),
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
    isModified_ = true;
    
    n_ = 0;
    p_ = 0;    
}

/**
   Set parameter n ( number of trials for a two-outcome experiment).
   This must be set before the actual generator is instantiated.
 */
void BinomialRng::setN(const Conn* c, double n)
{  
    BinomialRng* gen = static_cast<BinomialRng*>(c->data());
    ASSERT( n > 0 , "BinomialRng::setN(const Conn* c, double n) - n must be positive");    
    if ( gen )
    {
        gen->innerSetN(static_cast< unsigned long>( n ) );        
    }
    else
    {
        cerr << "ERROR: BinomialRng::innerSetN - object does not exist!" <<endl;
    }
}

void BinomialRng::innerSetN(unsigned long n)
{
    if ( n <= 0 )
    {
        cerr << "ERROR: BinomialRng::innerSetN - n must be a positive integer." << endl;
        return;
    }
    
    if(!isNSet_)
    {
        isNSet_ = true;
        n_ = n;
    }
    else 
    {
        if (n_!= n  )
        {
            n_ = n;
            isModified_ = true;            
        }
    }
    
    if ( isNSet_ && isPSet_ && isModified_)
    {   {
            if ( rng_ )
            {
                delete rng_;
            }           
            rng_ = new Binomial((unsigned long)n_,p_);
            isModified_ = false;            
        }     
    }
}

/**
   Returns parameter n.
 */
double BinomialRng::getN(Eref e)
{
    return (int)(static_cast <BinomialRng*>(e.data())->n_);
}

int BinomialRng::innerGetN()
{
    return n_;
}

/**
   Set parameter p ( the probability of the outcome of interest ).
   This must be set before the actual generator is instantiated.
 */
void BinomialRng::setP(const Conn* c, double p)
{
 
    
    BinomialRng* gen = static_cast<BinomialRng*>(c->data());
    if ( gen )
    {
        gen->innerSetP(p);        
    }
    else
    {
        cerr << "ERROR: BinomialRng::innerSetP - object does not exist!" <<endl;
    }
}

void BinomialRng::innerSetP(double p)
{
    
    if ( p < 0 || p > 1)
    {
        cerr << "ERROR: BinomialRng::setP - p must be in (0,1) range." << endl;
        return;
    }
    if ( !isPSet_)
    {
        p_ = p;
        isPSet_ = true;
    }
    else
    {
        if (!isEqual(p_,p))
        {
            p_ = p;
            isModified_ = true;            
        }
    }        
    
    if ( isNSet_ && isPSet_ && isModified_)
    {
        if ( rng_ )
        {
            delete rng_;            
        }
        rng_ = new Binomial((long)(n_),p_);
        isModified_ = false;        
    }
}

/**
   returns parameter p.
*/
double BinomialRng::getP(Eref e)
{
    return static_cast <BinomialRng*>(e.data())->p_;
}


double BinomialRng::innerGetP()
{
    return p_;
}

/**
   reports error if one or more of the parameters are not set.
*/
void BinomialRng::innerReinitFunc( const Conn* c, ProcInfo info)
{
    BinomialRng* gen = static_cast<BinomialRng*>(c->data());

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

double BinomialRng::innerGetNextSample()
{
    if(rng_)
    {
        return rng_->getNextSample();
    }
    else
    {
        cerr << "ERROR: You cannot ask for sample without setting n and p" << endl;        
    }
    return 0;    
}


#endif
