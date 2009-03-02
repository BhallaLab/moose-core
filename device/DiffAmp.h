// DiffAmp.h ---
// 
// Filename: DiffAmp.h
// Description: Difference amplifier.
// Author: subhasis ray
// Maintainer: 
// Created: Mon Dec 29 15:55:25 2008 (+0530)
// Version: 
// Last-Updated: Sun Mar  1 23:38:02 2009 (+0530)
//           By: subhasis ray
//     Update #: 25
// URL: 
// Keywords: 
// Compatibility: 
// 
// 

// Commentary: 
// This implementation follows the GENESIS version in logic.
// 
// 
// 

// Change log:
// 
// 
// 
// 
// 
/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2008 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU General Public License version 2
** See the file COPYING.LIB for the full notice.
**********************************************************************/

// Code:

#ifndef _DIFFAMP_H
#define _DIFFAMP_H

#include "basecode/header.h"
#include "basecode/moose.h"


class DiffAmp
{
  public:
    DiffAmp();
    
    static void setGain(const Conn* conn, double gain);
    static double getGain(Eref e);
    static void setSaturation(const Conn* conn, double saturation);
    static double getSaturation(Eref e);
    static double getOutput(Eref e);
    static double getPlus(Eref e);
    static double getMinus(Eref e);
    static void plusFunc(const Conn* c, double input);
    static void minusFunc(const Conn* c, double input);
    static void processFunc(const Conn* c, ProcInfo p);
    static void reinitFunc(const Conn* c, ProcInfo p);
    
  private:
    double gain_;
    double saturation_;
    double plus_;
    double minus_;
    double output_;
};
    
#endif

// 
// DiffAmp.h ends here
