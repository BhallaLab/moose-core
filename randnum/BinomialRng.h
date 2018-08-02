/*******************************************************************
 * File:            BinomialRng.h
 * Description:
 * Author:          Subhasis Ray
 * E-mail:          ray.subhasis@gmail.com
 * Created:         2007-11-08 10:48:59
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

#ifndef _BINOMIALRNG_H
#define _BINOMIALRNG_H

#include "RNG.h"
#include "../basecode/header.h"

class BinomialRng
{
public:
    BinomialRng();
    ~BinomialRng(){;}

    BinomialRng& operator=( const BinomialRng&);

    void setN(double n);
    double getN() const;

    void setP(double p);
    double getP() const;

    void reinitSeed( void );
    void reinit( const Eref& e, ProcPtr p);

    static const Cinfo * initCinfo();

private:
    unsigned long n_;
    double p_;

    unsigned long seed_;

    moose::MOOSE_RANDOM_DEVICE rd_;
    moose::MOOSE_RNG_DEFAULT_ENGINE rng_;
    moose::MOOSE_BINOMIAL_DISTRIBUTION dist_;

};

#endif
