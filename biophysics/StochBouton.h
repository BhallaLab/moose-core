/*******************************************************************
 * File:            StochBouton.h
 * Description:      
 * Author:          Subhasis Ray
 * E-mail:          ray.subhasis@gmail.com
 * Created:         2007-11-12 15:33:04
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

#ifndef _STOCHBOUTON_H
#define _STOCHBOUTON_H
#include "header.h"
#include "moose.h"

class StochBouton
{
  public:
    StochBouton();
    
    static void channelFunc(const Conn& c, double Vm);
    static void setPoolSize(const Conn& c, double poolSize);
    static void setVesicleP(const Conn& c, double vesicleP);
    static double getPoolSize(const Element * e);
    static double getVesicleP(const Element* e);
    static void setReleaseCount(const Conn& c, double count);
    static void incrementPool(const Conn& c, double count);    
    static void processFunc(const Conn& c, ProcInfo p);
    static void reinitFunc(const Conn& c, ProcInfo p);
    
    
  private:
    double Vm_;
    /// Number of vesicles released within a small time window.
    double releaseCount_;
    /// Probability of an individual vesicle being released.
    double vesicleP_;
    /// Number of ready to release vesicles
    double poolSize_;
};

#endif
