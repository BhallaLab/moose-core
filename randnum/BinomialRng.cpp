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

#include "../basecode/global.h"
#include "utility/numutil.h"
#include "BinomialRng.h"
#include "RNG.h"

#include <cmath>

extern const Cinfo* initRandGeneratorCinfo();

const Cinfo* BinomialRng::initCinfo()
{
    static ValueFinfo< BinomialRng, double > n(
        "n",
        "Parameter n of the binomial distribution. In a coin toss experiment,"
        " this is the number of tosses.",
        &BinomialRng::setN,
        &BinomialRng::getN
        );

    static ValueFinfo < BinomialRng, double > p(
        "p",
        "Parameter p of the binomial distribution. In a coin toss experiment,"
        " this is the probability of one of the two sides of the coin being on"
        " top.",
        &BinomialRng::setP,
        &BinomialRng::getP
        );

    static Finfo* binomialRngFinfos[] =
    {
        &n,
        &p,
    };

    static string doc[] =
    {
        "Name", "BinomialRng",
        "Author", "Subhasis Ray, Dilawar Singh",
        "Description", "Binomially distributed random number generator.",
    };

    Dinfo < BinomialRng> dinfo;

    static Cinfo binomialRngCinfo(
        "BinomialRng",
        Neutral::initCinfo(),
        binomialRngFinfos,
        sizeof(binomialRngFinfos)/sizeof(Finfo*),
        &dinfo,
        doc,
        sizeof( doc ) / sizeof( string ));
    return &binomialRngCinfo;
}

static const Cinfo* binomialRngCinfo = BinomialRng::initCinfo();


BinomialRng::BinomialRng()
{
    n_ = 0;
    p_ = 0;
    dist_ = moose::MOOSE_BINOMIAL_DISTRIBUTION( n_, p_ );
}

BinomialRng& BinomialRng::operator=( const BinomialRng&)
{
    return *this;
}

/**
   Set parameter n ( number of trials for a two-outcome experiment).
   This must be set before the actual generator is instantiated.
 */
void BinomialRng::setN(double value)
{
    n_ = (unsigned long)value;
    dist_ = moose::MOOSE_BINOMIAL_DISTRIBUTION( n_, p_ );
}

/**
   Returns parameter n.
 */
double BinomialRng::getN() const
{
    return n_;
}

/**
   Set parameter p ( the probability of the outcome of interest ).
   This must be set before the actual generator is instantiated.
 */
void BinomialRng::setP(double p)
{
    if ( p < 0 || p > 1)
    {
        cerr << "ERROR: BinomialRng::setP - p must be in (0,1) range." << endl;
        return;
    }
    p_ = p;
    dist_ = moose::MOOSE_BINOMIAL_DISTRIBUTION( n_, p_ );
}

/**
   returns parameter p.
*/
double BinomialRng::getP() const
{
    return p_;
}

void BinomialRng::reinitSeed()
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
    dist_ = moose::MOOSE_BINOMIAL_DISTRIBUTION( n_, p_ );
}

/**
   reports error if one or more of the parameters are not set.
*/
void BinomialRng::reinit( const Eref& e, ProcPtr p)
{
    reinitSeed( );
    dist_ = moose::MOOSE_BINOMIAL_DISTRIBUTION( n_, p_ );
}

