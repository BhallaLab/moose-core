// RC.h ---
// 
// Filename: RC.h
// Description: 
// Author: subhasis ray
// Maintainer: 
// Created: Wed Dec 31 15:18:22 2008 (+0530)
// Version: 
// Last-Updated: Sat Jan  3 20:09:14 2009 (+0530)
//           By: subhasis ray
//     Update #: 22
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

#ifndef _RC_H
#define _RC_H

#include "basecode/header.h"
#include "basecode/moose.h"

class RC{
  public:
    RC();
    
    static void setV0( const Conn& conn, double voltage );
    static double getV0( Eref e );
    static void setResistance( const Conn& conn, double resistance );
    static double getResistance( Eref e );
    static void setCapacitance( const Conn& conn, double capacitance );
    static double getCapacitance( Eref e );
    static double getState( Eref e );
    static void setInject( const Conn& conn, double inject );
    static double getInject( Eref e );
    static void processFunc( const Conn& conn, ProcInfo proc );
    static void reinitFunc( const Conn& conn, ProcInfo proc );
    
  private:
    double v0_;
    double resistance_;
    double capacitance_;
    double state_;
    double inject_;
    double inject_prev_;
    double exp_;
    double dt_;
    int isteps_;
};

#endif


// 
// RC.h ends here
