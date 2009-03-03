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
#include <cmath>
extern const Cinfo* initRandGeneratorCinfo();

const Cinfo* initGammaRngCinfo()
{
    static Finfo* gammaRngFinfos[] =
        {
            new ValueFinfo("alpha", ValueFtype1<double>::global(),
                           GFCAST( &GammaRng::getAlpha),
                           RFCAST( &GammaRng::setAlpha)),
            new ValueFinfo("theta", ValueFtype1<double>::global(),
                            GFCAST(&GammaRng::getTheta),
                            RFCAST(&GammaRng::setTheta)),
        };
    
	static string doc[] =
	{
		"Name", "GammaRng",
		"Author", "Subhasis Ray",
		"Description", "Gamma distributed random number generator.",
	};
    
	static Cinfo gammaRngCinfo(
                                doc,
				sizeof( doc ) / sizeof( string ),
                                initRandGeneratorCinfo(),
                                gammaRngFinfos,
                                sizeof(gammaRngFinfos)/sizeof(Finfo*),
                                ValueFtype1<GammaRng>::global()
                                );
    return &gammaRngCinfo;
}

    
static const Cinfo* gammaRngCinfo = initGammaRngCinfo();

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
double GammaRng::getAlpha(Eref e)
{
    return static_cast<GammaRng*> (e.data())->alpha_;    
}
/**
   Sets parameter alpha. Also known as the shape parameter.
*/
void GammaRng::setAlpha(const Conn* c, double alpha)
{
    
    if (fabs(alpha) < DBL_MIN)
    {
        cerr << "ERROR: Shape parameter alpha must be non-zero." << endl;
        return;
    }
    GammaRng* generator = static_cast<GammaRng*>(c->data());
    if ( generator->rng_ )
    {
        generator->alpha_ = static_cast<Gamma*>(generator->rng_)->getAlpha();
        return;        
    }
    generator->alpha_ = alpha;
    generator->isAlphaSet_ = true;
    if ( generator->isThetaSet_ )
    {
        generator->rng_ = new Gamma(generator->alpha_, generator->theta_);
    }
}
/**
   returns the scale parameter.
*/
double GammaRng::getTheta(Eref e)
{
    return static_cast<GammaRng*> (e.data())->theta_;    
}

/**
   Sets parameter theta. Also known as the scale parameter.
*/
void GammaRng::setTheta(const Conn* c, double theta)
{
    
    if (fabs(theta) < DBL_MIN)
    {
        cerr << "ERROR: Scale parameter theta must be non-zero." << endl;
        return;
    }
    GammaRng* generator = static_cast<GammaRng*>(c->data());
    if ( generator->rng_ )
    {
        generator->theta_ = static_cast<Gamma*>(generator->rng_)->getTheta();
        return;        
    }    
    generator->theta_ = theta;
    generator->isThetaSet_ = true;
    if ( generator->isAlphaSet_ )
    {
        generator->rng_ = new Gamma(generator->alpha_, generator->theta_);
    }
}
/**
   reports error if parameters have not been set properly.
*/
void GammaRng::innerReinitFunc(const Conn* c, ProcInfo info)
{
    GammaRng* generator = static_cast < GammaRng* >(c->data());
    if (! generator->rng_ )
    {
        cerr << "ERROR: GammaRng::innerReinitFunc - parameters alpha and theta must be set before using the Gamma distribution generator." << endl;                
    }
}

#endif
