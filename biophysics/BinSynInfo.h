/*******************************************************************
 * File:            BinSynInfo.h
 * Description:      Extends SynInfo by adding the binomial
 *                      distribution parameters n and p.
 * Author:          Subhasis Ray
 * E-mail:          ray.subhasis@gmail.com
 * Created:         2007-11-26 13:55:57
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

#ifndef _BINSYNINFO_H
#define _BINSYNINFO_H
#include "SynInfo.h"
#include "BinomialRng.h"
/**
   SynInfo extended for incorporating binomial rng.
   Partly copied from SynInfo.h
 */
class BinSynInfo:public SynInfo
{
  public:
    BinSynInfo():SynInfo()
    {
        ;
    }
    BinSynInfo(double w, double d):SynInfo(w,d)
    {
        ;
    }
    BinSynInfo(double w, double d, double n, double p):SynInfo(w,d)
    {
        ;
    }
    		// This is backward because the syntax of the priority
		// queue puts the _largest_ element on top.
    bool operator< ( const BinSynInfo& other ) const {
        return delay > other.delay;
    }
    
    bool operator== ( const BinSynInfo& other ) const {
        return delay == other.delay && weight == other.weight;
    }
    
    SynInfo event( double time ) {
        releaseCount = rng.innerGetNextSample();
        return SynInfo( weight*releaseCount, time + delay );
    }
    void setPoolSize(int n)
    {
        rng.innerSetN(n);
    }
    int getPoolSize()
    {
        return rng.innerGetN();
    }
    void setReleaseP(double p)
    {
        rng.innerSetP(p);
    }
    double getReleaseP()
    {
        return rng.innerGetP();
    }
    double getReleaseCount()
    {
        return releaseCount;
    }
    
  private:
    double releaseCount;    
    BinomialRng rng;    
};

#endif

