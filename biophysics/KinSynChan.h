// 
// Filename: KinSynChan.h
// Description: 
// Author: Aditya Gilra
// Maintainer: 
// Created: Mon Apr 26
// Version: 
// Last-Updated: Tue Apr 27 23:31:58 2010 (+0530)
//           By: Subhasis Ray
//     Update #: 58
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

#ifndef _KinSynChan_h
#define _KinSynChan_h

#include "SynChan.h"
class KinSynChan: public SynChan
{
  public:
    KinSynChan();
    static void setRinf(const Conn* conn, double value);
    static double getRinf(Eref e);
    static void setPulseWidth(const Conn* conn, double value);
    static double getPulseWidth(Eref e);
    
    static void setTau1(const Conn* conn, double value); // decay time constant after pulse is over
    static double getTau1(Eref e);
    //static void setTau2(const Conn* conn, double value);
    //static double getTau2(Eref e);
  protected:
    void innerSetRinf(double value);
    double innerGetRinf();
    double innerGetTau1();
    void innerSetTau1(double value);
    //double innerGetTau2();
    //void innerSetTau2(double value);
    void innerSetPulseWidth(double value);
    double innerGetPulseWidth();
    void innerProcessFunc(Eref e, ProcInfo info);
    void innerReinitFunc(Eref e, ProcInfo info);
    
  protected:
    double rInf_; // r_infinty in Destexhe, Mainen, Sejnowski paper.
    double pulseWidth_; // t1 in  Destexhe, Mainen, Sejnowski paper.
    double tau1_; // tau1 in Destexhe, Mainen, Sejnowski paper.
    double tau2_; // tau2=tau_r = rise time constant in Destexhe, Mainen, Sejnowski paper. It is calculated internally from tau1 and rInf.
};

#endif
// 
// KinSynChan.h ends here
