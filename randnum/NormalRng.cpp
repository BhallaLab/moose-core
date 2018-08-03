/*******************************************************************
 * File:            NormalRng.cpp
 * Description:      This is the MOOSE front end for class Normal,
 *                   which generates normally distributed random
 *                   doubles.
 * Author:          Subhasis Ray
 * E-mail:          ray.subhasis@gmail.com
 * Created:         2007-11-03 22:07:04
 * LOGS:
 *  Thursday 02 August 2018 05:04:15 PM IST, Dilawar Singh
 *      - use <random> or boost to generate the distributions.
 ********************************************************************/

#include "NormalRng.h"
#include "../basecode/global.h"

const Cinfo* NormalRng::initCinfo()
{
    static ValueFinfo< NormalRng, double > mean(
        "mean",
        "Mean of the normal distribution",
        &NormalRng::setMean,
        &NormalRng::getMean
    );

    static ValueFinfo< NormalRng, unsigned long > seed(
        "seed",
        "Set the seed of random number generator.",
        &NormalRng::setSeed,
        &NormalRng::getSeed
    );

    static ValueFinfo< NormalRng, double > variance(
        "variance",
        "Variance of the normal distribution",
        &NormalRng::setVariance,
        &NormalRng::getVariance
    );

    static DestFinfo reinit( "reinit",
        "Handles reinit call from Clock"
        , new ProcOpFunc< NormalRng >( &NormalRng::reinit )
    );

    static Finfo* normalRngFinfos[] =
    {
        &mean,
        &seed,
        &variance
    };

    static string doc[] =
    {
        "Name", "NormalRng",
        "Author", "Subhasis Ray, Dilawar Singh",
        "Description", "Normally distributed random number generator.",
    };

    Dinfo< NormalRng > dinfo;

    static Cinfo normalRngCinfo(
        "NormalRng",
        Neutral::initCinfo(),
        normalRngFinfos,
        sizeof(normalRngFinfos)/sizeof(Finfo*),
        &dinfo,
        doc,
        sizeof( doc ) / sizeof( string )
    );

    return &normalRngCinfo;
}


static const Cinfo* normalRngCinfo = NormalRng::initCinfo();

NormalRng& NormalRng::operator=( const NormalRng& rng )
{
    return *this;
}

/** Set the mean of the internal generator object.  */
void NormalRng::setMean(double mean)
{
    mean_ = mean;
    dist_ = moose::MOOSE_NORMAL_DISTRIBUTION(mean, variance_);
}

double NormalRng::getMean( void ) const
{
    return mean_;
}

/** Set the seed for random number generator.  */
void NormalRng::setSeed(unsigned long seed)
{
    seed_ = seed;
    rng_.seed( seed_ );
}

unsigned long NormalRng::getSeed( void ) const
{
    return seed_;
}


/**
   Since normal distribution is defined in terms of mean and variance, we
   want to store them in order to create the internal generator object.
 */
void NormalRng::setVariance(double variance)
{
    if ( variance < 0 )
    {
        cerr << "ERROR: variance cannot be negative." << endl;
        return;
    }
    cerr << "Info: Variance is " << variance << endl;
    variance_ = variance;
    dist_ = moose::MOOSE_NORMAL_DISTRIBUTION(mean_, variance);
}

double NormalRng::getVariance( void ) const
{
    return variance_;
}

void NormalRng::reinitSeed( void )
{
    if( seed_ > 0 )
    {
        rng_.seed( seed_ );
        return;
    }

    if( moose::getGlobalSeed() > 0 )
    {
        rng_.seed( moose::getGlobalSeed() );
        return;
    }

    rng_.seed( rd_() );
}

void NormalRng::reinit(const Eref& e, ProcPtr p)
{
    // Just in case; to be safe.
    reinitSeed();
    dist_ = moose::MOOSE_NORMAL_DISTRIBUTION(mean_, variance_);
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
NormalRng::NormalRng()
{
    mean_ = 0.0;
    variance_ = 1.0;
    seed_ = moose::getGlobalSeed( );

    if( seed_ > 0 )
        rng_.seed( seed_ );
    else
        rng_.seed( rd_() );
}

double NormalRng::getNextSample()
{
    return dist_( rng_ );
}


