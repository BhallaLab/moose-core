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

#include "NormalRng.h"

const Cinfo* NormalRng::initCinfo()
{
    static ValueFinfo< NormalRng, double > mean(
        "mean",
        "Mean of the normal distribution",
        &NormalRng::setMean,
        &NormalRng::getMean
    );

    static ValueFinfo< NormalRng, double > variance(
        "variance",
        "Variance of the normal distribution",
        &NormalRng::setVariance,
        &NormalRng::getVariance
    );

    static Finfo* normalRngFinfos[] =
    {
        &mean,
        &variance
    };

    static string doc[] =
    {
        "Name", "NormalRng",
        "Author", "Subhasis Ray",
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

/**
   Set the mean of the internal generator object.
 */
void NormalRng::setMean(double mean)
{
    mean_ = mean;
    dist_ = moose::MOOSE_NORMAL_DISTRIBUTION(mean, variance_);
}

double NormalRng::getMean( void ) const
{
    return mean_;
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
    variance_ = variance;
    dist_ = moose::MOOSE_NORMAL_DISTRIBUTION(mean_, variance);
}

double NormalRng::getVariance( void ) const
{
    return variance_;
}

void NormalRng::vReinit(const Eref& e, ProcPtr p)
{
    // Just in case; to be safe.
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
}

double NormalRng::getNextSample()
{
    return dist_( rng_ );
}


