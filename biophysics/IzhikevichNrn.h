// IzhikevichNrn.h --- 

// 
// Filename: IzhikevichNrn.h
// Description: 
// Author: subhasis ray
// Maintainer: 
// Created: Fri Apr  3 17:52:55 2009 (+0530)
// Version: 
// Last-Updated: Wed Jun 23 17:43:27 2010 (+0530)
//           By: Subhasis Ray
//     Update #: 34
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

#ifndef _IZHIKEVICHNRN_H
#define _IZHIKEVICHNRN_H

class IzhikevichNrn
{
  public:
    IzhikevichNrn();
    static void setA(const Conn* conn, double value);
    static double getA(Eref e);
    static void setB(const Conn* conn, double value);
    static double getB(Eref e);
    static void setC(const Conn* conn, double value);
    static double getC(Eref e);
    static void setD(const Conn* conn, double value);
    static double getD(Eref e);
    static void setVmax(const Conn* conn, double value);
    static double getVmax(Eref e);
    static void setAlpha(const Conn* conn, double value);
    static double getAlpha(Eref e);
    static void setBeta(const Conn* conn, double value);
    static double getBeta(Eref e);
    static void setGamma(const Conn* conn, double value);
    static double getGamma(Eref e);
    static void setInject(const Conn* conn, double value);
    static double getInject(Eref e);
    static void setVm(const Conn* conn, double value);
    static double getVm(Eref e);
    static void setInitVm(const Conn* conn, double value);
    static double getInitVm(Eref e);
    static void setInitU(const Conn* conn, double value);
    static double getInitU(Eref e);
    static double getU(Eref e);
    static void processFunc(const Conn* conn, ProcInfo proc);
    static void reinitFunc(const Conn* conn, ProcInfo proc);
    static void setRm(const Conn * conn, double value);
    static double getRm(Eref e);
    
  private:
    double alpha_;
    double beta_;
    double gamma_;
    double Rm_;
    double a_;
    double b_;
    double c_;
    double d_;
    double Vm_;
    double u_;
    double Vmax_;
    double initVm_;
    double initU_;
    double sum_inject_;
    double Im_;
    double savedVm_;
};
#endif



// 
// IzhikevichNrn.h ends here
