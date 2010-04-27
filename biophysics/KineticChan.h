// 
// Filename: KineticChan.h
// Description: 
// Author: Aditya Gilra
// Maintainer: 
// Created: Mon Apr 26
// Version: 
// Last-Updated: Mon Apr 26
//           By: Aditya Gilra
//     Update #: --
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

#ifndef _KineticChan_h
#define _KineticChan_h

#include "SynChan.h"

class KineticChan: public SynChan
{
  public:
    KineticChan();
    static void setTrPulseTime(const Conn* conn, double conc);
    void innerSetTrPulseTime(double value);
    static double getTrPulseTime(Eref e);
    double innerGetTrPulseTime();
  protected:
    double TrPulseTime_;

};

#endif
// 
// NMDAChan.h ends here
