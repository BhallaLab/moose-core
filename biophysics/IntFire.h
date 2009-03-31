// 
// Filename: IntFire.h
// Description: 
// Author: subhasis ray
// Maintainer: 
// Created: Thu Mar 19 18:52:52 2009 (+0530)
// Version: 
// Last-Updated: Wed Mar 25 16:38:25 2009 (+0530)
//           By: subhasis ray
//     Update #: 67
// URL: 
// Keywords: 
// Compatibility: 
// 
// 

// Commentary:
//
//      This is just a refinement of IntFire.h by Upi in MSG branch of
//  MOOSE svn. Defines a single compartment leaky integrate and fire
//  neuron.
// 
// 
// 

// Change log:
// 2009-03-19 18:54:52 (+0530) Initial port to main code.
// 
// 
// 
/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

// Code:


#ifndef   	INTFIRE_H_
#define   	INTFIRE_H_

class IntFire
{
  public:
    IntFire();
    static void setVr(const Conn* c, double Vr); /// Set reset voltage
    static double getVr(Eref e); /// get reset voltage
    static void setVt(const Conn* c, double Vt); /// set threshold voltage
    static double getVt(Eref e); /// get threshold voltage
    //    static void setTau(const Conn* c, double tau); /// set time constant
    static double getTau(Eref e); /// get time constant
    static double getInitVm(Eref e);
    static void setInitVm(const Conn* c, double initVm);
    static double getCm(Eref e);
    static void setCm(const Conn* c, double Cm);
    static double getRm(Eref e);
    static void setRm(const Conn* c, double Rm);
    static double getVm(Eref e);
    static void setVm(const Conn* c, double Vm);
    static double getEm(Eref e);
    static void setEm(const Conn* c, double Vm);
    static double getInject(Eref e);
    static void setInject(const Conn* c, double inject);
    static double getRefractT(Eref e);
    static void setRefractT(const Conn* c, double refractT);
    static void injectDestFunc(const Conn* c, double inject);
    static void processFunc(const Conn* c, ProcInfo p);
    static void reinitFunc(const Conn* c, ProcInfo p);
    void innerProcessFunc(Eref e, ProcInfo p);
    void innerReinitFunc(Eref e, ProcInfo p);
    static void channelFunc(const Conn* c, double Gk, double Ek);
    static const double EPSILON;
  private:
    double initVm_; // initial membrane potential
    double Vt_; // threshold
    double Vr_; // reset
    double Vm_; // membrane potential - the state variable
    double tau_; // time constant
    double Rm_; // leak resistance
    double Cm_;
    double refractT_; // absolute refractory period
    double Em_; // reversal potential for leak current
    double A_; // internal variable for intermediate results
    double B_; // internal variable for intermediate results
    // TODO: Extend this for multicompartmental IF with Ra?
    double lastEvent_;
    double inject_;
    double sumInject_;
    double Im_;
};

#endif 	    /* !INTFIRE_H_ */




// 
// IntFire.h ends here
