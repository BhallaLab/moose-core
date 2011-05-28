/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2011 Upinder S. Bhalla. and NCBS
** This program is free software; you can redistribute it and/or
** modify it under the terms of the GNU General Public License as
** published by the Free Software Foundation; either version 3, or
** (at your option) any later version.
** 
** This program is distributed in the hope that it will be useful,
** but WITHOUT ANY WARRANTY; without even the implied warranty of
** MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
** General Public License for more details.
** 
** You should have received a copy of the GNU General Public License
** along with this program; see the file COPYING.  If not, write to
** the Free Software Foundation, Inc., 51 Franklin Street, Fifth
** Floor, Boston, MA 02110-1301, USA.
**********************************************************************/
// 
// Filename: NMDAChan.h
// Description: 
// Author: Subhasis Ray
// Maintainer: 
// Created: Mon Mar  8 15:13:02 2010 (+0530)
// Version: 
// Last-Updated: Sat May 28 12:06:46 2011 (+0530)
//           By: Subhasis Ray
//     Update #: 83
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
// 

// Code:

#ifndef _NMDAChan_h
#define _NMDAChan_h

class NMDAChan: public SynChan
{
  public:
    NMDAChan();
    void setTransitionParam( unsigned int index, double val );
    double getTransitionParam( unsigned int index) const;
    double getUnblocked() const;
    double getSaturation() const;
    void setSaturation( double value);
    void process( const Eref& e, ProcPtr info);
    void reinit( const Eref& e, ProcPtr info);
    void setMgConc( double conc );
    double getMgConc() const;
    unsigned int updateNumSynapse() const;
    /////////////////////////////////////////////////////////////
    static const Cinfo* initCinfo();
    
  protected:
    vector< double > c_;
    double A_;
    double B1_;
    double B2_;
    double x_;
    double y_;
    double Mg_;
    double unblocked_;
    double saturation_;
    priority_queue<Synapse> oldEvents_; // This is for removing the
                                         // effects of old event after
                                         // tau1
    double decayFactor_; // Intermediate variable for Exponential
                         // Euler method exp(-t/tau2)
};

#endif
// 
// NMDAChan.h ends here
