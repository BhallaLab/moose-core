// 
// Filename: NMDAChan.h
// Description: 
// Author: Subhasis Ray
// Maintainer: 
// Created: Mon Mar  8 15:13:02 2010 (+0530)
// Version: 
// Last-Updated: Thu Mar 11 14:42:18 2010 (+0530)
//           By: Subhasis Ray
//     Update #: 63
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

#ifndef _NMDACchan_h

#include "SynChan.h"

class NMDAChan: public SynChan
{
  public:
    NMDAChan();
    static void setTransitionParam(const Conn* c, int index, double value);
    void innerSetTransitionParam(int index, double value);
    static double getTransitionParam(Eref e, int index);
    double innerGetTransitionParam(int index);
    static double getUnblocked(Eref e);
    double innerGetUnblocked();
    static void processFunc(const Conn* conn, ProcInfo info);
    void innerProcessFunc(Eref e, ProcInfo info);
    static void reinitFunc(const Conn* conn, ProcInfo info);
    void innerReinitFunc(Eref e, ProcInfo info);
    static void setMgConc(const Conn* conn, double conc);
    void innerSetMgConc(double value);
    static double getMgConc(Eref e);
    double innerGetMgConc();
    
  protected:
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
    double x_;
    double y_;
    double Mg_;
    double unblocked_;
    priority_queue<SynInfo> oldEvents_; // This is for removing the
                                         // effects of old event after tau1
};

#endif
// 
// NMDAChan.h ends here
