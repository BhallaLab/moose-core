/*******************************************************************
 * File:            PoissonRng.h
 * Description:
 * Author:          Subhasis Ray
 * E-mail:          ray.subhasis@gmail.com
 * Created:         2007-11-07 16:22:35
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

#ifndef _POISSONRNG_H
#define _POISSONRNG_H

class PoissonRng
{
public:

    PoissonRng();
    ~PoissonRng() { ; }

    PoissonRng& operator=(const PoissonRng& );

    void setMean(double mean);
    double getMean( ) const;

    void reinitSeed( void );
    void reinit(const Eref& e, ProcPtr p);

    static const Cinfo * initCinfo();

private:

    double mean_;

    moose::MOOSE_RANDOM_DEVICE rd_;
    moose::MOOSE_POISSON_DISTRIBUTION dist_;
    moose::MOOSE_RNG_DEFAULT_ENGINE rng_;
};


#endif
