// PIDController.h --- 
// 
// Filename: PIDController.h
// Description: 
// Author: subhasis ray
// Maintainer: 
// Created: Tue Dec 30 23:14:00 2008 (+0530)
// Version: 
// Last-Updated: Mon Mar  2 01:27:53 2009 (+0530)
//           By: subhasis ray
//     Update #: 45
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

#ifndef _PIDCONTROLLER_H
#define _PIDCONTROLLER_H

#include "basecode/header.h"
#include "basecode/moose.h"

class PIDController{
  public:
    PIDController();
    
    static void setCommand( const Conn* conn, double command );
    static double getCommand( Eref e );
    static void setSensed( const Conn* conn, double sensed );
    static double getSensed( Eref e );
    static double getOutput( Eref e );
    static void setGain( const Conn* conn, double gain );
    static double getGain( Eref e );
    static void setTauI( const Conn* conn, double tau_i );
    static double getTauI( Eref e );
    static void setTauD( const Conn* conn, double tau_d );
    static double getTauD( Eref e );
    static void setSaturation( const Conn* conn, double saturation );
    static double getSaturation( Eref e );
    static double getError( Eref e );
    static double getEIntegral( Eref e );
    static double getEDerivative( Eref e );
    static double getEPrevious( Eref e );
    static void processFunc( const Conn* conn, ProcInfo process );
    static void reinitFunc( const Conn* conn, ProcInfo process );
    
  private:
    double command_;
    double saturation_;
    double gain_;
    double tau_i_;
    double tau_d_;
    double sensed_;    
    double output_;
    double error_; // e of PIDController in GENESIS ( error = command - sensed )
    double e_integral_; // integral of error dt
    double e_derivative_; // derivative of error 
    double e_previous_;
};
  
#endif

// Code:




// 
// PIDController.h ends here
