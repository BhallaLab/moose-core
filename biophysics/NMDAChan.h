// 
// Filename: NMDAChan.h
// Description: 
// Author: Subhasis Ray
// Maintainer: 
// Created: Mon Mar  8 15:13:02 2010 (+0530)
// Version: 
// Last-Updated: Wed Jun 22 10:49:16 2011 (+0530)
//           By: Subhasis Ray
//     Update #: 86
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
// 
// 
// 
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 3, or
// (at your option) any later version.
// 
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// General Public License for more details.
// 
// You should have received a copy of the GNU General Public License
// along with this program; see the file COPYING.  If not, write to
// the Free Software Foundation, Inc., 51 Franklin Street, Fifth
// Floor, Boston, MA 02110-1301, USA.
// 
// 

// Code:

#ifndef _NMDAChan_h
#define _NMDAChan_h

#include "SynChan.h"

class NMDAChan: public SynChan
{
  public:
    NMDAChan();
    static void setTransitionParam(const Conn* c, double value, const unsigned int& index);
    static double getTransitionParam(Eref e, const unsigned int& index);
    static double getUnblocked(Eref e);
    static double getSaturation(Eref e);
    static void setSaturation(const Conn * conn, double value);
    static void processFunc(const Conn* conn, ProcInfo info);
    static void reinitFunc(const Conn* conn, ProcInfo info);
    static void setMgConc(const Conn* conn, double conc);
    static double getMgConc(Eref e);
    
  protected:
    virtual void innerSynapseFunc( const Conn* c, double time );
    virtual unsigned int updateNumSynapse( Eref e );
    void innerSetTransitionParam(double value, const unsigned int index);
    double innerGetTransitionParam(unsigned int index);
    double innerGetUnblocked();
    double innerGetSaturation();
    void innerSetSaturation(double value);
    virtual void innerProcessFunc(Eref e, ProcInfo info);
    virtual void innerReinitFunc(Eref e, ProcInfo info);
    void innerSetMgConc(double value);
    double innerGetMgConc();
    
    double c0_;
    double c1_;
    double c2_;
    double c3_;
    double c4_;
    double c5_;
    double c6_;
    double c7_;
    double c8_;
    double c9_;
    double c10_;
    double A_;
    double B1_;
    double B2_;
    // double x_;
    // double y_;
    double Mg_;
    double unblocked_;
    double saturation_;
    priority_queue<SynInfo> oldEvents_; // This is for removing the
                                         // effects of old event after
                                         // tau1
    double decayFactor_; // Intermediate variable for Exponential
                         // Euler method exp(-t/tau2)
};

#endif
// 
// NMDAChan.h ends here
