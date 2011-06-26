// STPNMDAChan.h --- 
// 
// Filename: STPNMDAChan.h
// Description: 
// Author: Subhasis Ray
// Maintainer: 
// Copyright (C) 2010 Subhasis Ray, all rights reserved.
// Created: Sat Jun 25 15:06:52 2011 (+0530)
// Version: 
// Last-Updated: Sat Jun 25 15:46:32 2011 (+0530)
//           By: Subhasis Ray
//     Update #: 28
// URL: 
// Keywords: 
// Compatibility: 
// 
// 

// Commentary: 
// 
// 
// 
// 

// Change log:
// 
// 2011-06-25 15:07:17 (+0530) Started from scratch after the old code
// turned out to be too much to debug.
// 

// Code:

#ifndef _STPNMDACHAN_H
#define _STPNMDACHAN_H

#include "moose.h"
#include "header.h"

#include "STPSynChan.h"

class STPNMDAChan: public STPSynChan
{
  public:
    STPNMDAChan();
    virtual ~STPNMDAChan();    
    static void setMgConc(const Conn * conn, double value);
    static double getMgConc(Eref e);
    static void setTransitionParam(const Conn * conn, double value, const unsigned int & index);
    static double getTransitionParam(Eref e, const unsigned int& index);
    static void setSaturation(const Conn * conn, double value);
    static double getSaturation(Eref e);
    static double getUnblocked(Eref e);
  protected:
    virtual void innerSynapseFunc(const Conn * conn, double time);
    virtual void innerProcessFunc(Eref e, ProcInfo p);
    virtual void innerReinitFunc( Eref e,  ProcInfo p );
    void innerSetTransitionParam(Eref e, double value, const unsigned int index);
    double innerGetTransitionParam(Eref e, unsigned int index);
    double innerGetUnblocked();
    double innerGetSaturation();
    void innerSetSaturation(double value);
    void innerSetMgConc(double value);
    double innerGetMgConc();
    /////////////////////////////////////
    // Fields
    /////////////////////////////////////
    double saturation_, unblocked_, MgConc_, A_, B1_, B2_, decayFactor_;
    vector<double> c_;
    priority_queue<SynInfo> oldEvents_;
};

extern const Cinfo * initSTPSynChanCinfo();


#endif


// 
// STPNMDAChan.h ends here
