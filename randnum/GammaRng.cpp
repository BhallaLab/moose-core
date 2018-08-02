/*******************************************************************
 * File:            GammaRng.cpp
 * Description:
 * Author:          Subhasis Ray
 * E-mail:          ray.subhasis@gmail.com
 * Created:         2007-11-08 11:56:00
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

#ifndef _GAMMARNG_CPP
#define _GAMMARNG_CPP
#include "GammaRng.h"
#include "utility/numutil.h"
#include <cmath>

const Cinfo* GammaRng::initCinfo()
{
    static ValueFinfo< GammaRng, double > alpha(
        "alpha",
        "Parameter alpha of the gamma distribution.",
        &GammaRng::setAlpha,
        &GammaRng::getAlpha);
    static ValueFinfo< GammaRng, double > theta(
        "theta",
        "Parameter theta of the Gamma distribution.",
        &GammaRng::setTheta,
        &GammaRng::getTheta);
    static Finfo* gammaRngFinfos[] = {
        &alpha,
        &theta,
    };

    static string doc[] = {
        "Name", "GammaRng",
        "Author", "Subhasis Ray",
        "Description", "Gamma distributed random number generator.",
    };

    Dinfo < GammaRng > dinfo;
    static Cinfo gammaRngCinfo(
        "GammaRng",
        RandGenerator::initCinfo(),
        gammaRngFinfos,
        sizeof(gammaRngFinfos)/sizeof(Finfo*),
        &dinfo,
        doc,
        sizeof( doc ) / sizeof( string ));
    return &gammaRngCinfo;
}


static const Cinfo* gammaRngCinfo = GammaRng::initCinfo();

GammaRng::GammaRng()
{
    isAlphaSet_ = false;
    isThetaSet_ = false;
    alpha_ = 1;
    theta_ = 1;
}
/**
   returns the shape parameter.
*/
double GammaRng::getAlpha() const
{
    return alpha_;
}
/**
   Sets parameter alpha. Also known as the shape parameter.
*/
void GammaRng::setAlpha(double alpha)
{
    alpha_ = alpha;
}
/**
   returns the scale parameter.
*/
double GammaRng::getTheta()const
{
    return theta_;
}

/**
   Sets parameter theta. Also known as the scale parameter.
*/
void GammaRng::setTheta(double theta)
{
    theta_ = theta;
}

/**
   reports error if parameters have not been set properly.
*/
void GammaRng::vReinit(const Eref& e, ProcPtr p)
{
    if (! rng_ )
    {
        cerr << "ERROR: GammaRng::vReinit - parameters alpha and theta must be set before using the Gamma distribution generator." << endl;
    }
}

#endif
