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

#include "randnum.h"
#include "basecode/header.h"
#include "basecode/moose.h"
#include "RandGenerator.h"
#include "Binomial.h"

class BinomialRng: public RandGenerator
{
  public:
    BinomialRng();
    void innerSetN(unsigned long n);
    int innerGetN();
    void innerSetP(double p);
    double innerGetP();
    double innerGetNextSample();
    
    static void setN(const Conn* c, double n);    
    static double getN(Eref e);    
    static void setP(const Conn* c, double p);
    static double getP(Eref e);    
    virtual void innerReinitFunc( const Conn* c, ProcInfo info);

  private:
    bool isNSet_;
    unsigned long n_;
    bool isPSet_;
    double p_;
    bool isModified_;    
};

#endif
