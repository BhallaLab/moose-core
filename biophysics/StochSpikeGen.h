// 
// Filename: StochSpikeGen.h
// Description: 
// Author: Subhasis Ray
// Maintainer: 
// Copyright (C) 2010 Subhasis Ray, all rights reserved.
// Created: Mon Dec 13 15:54:16 2010 (+0530)
// Version: 
// Last-Updated: Mon May  2 11:35:27 2011 (+0530)
//           By: Subhasis Ray
//     Update #: 48
// URL: 
// Keywords: 
// Compatibility: 
// 
// 

// Commentary: 
// 
// StochSpikeGen is the stochastic Spike Generator. The idea is to
// implement simple stochasticity in presynaptic release. This is
// based on Stevens and Wang, 1995:
//
// "For 16 neurons, the initial success probability (the probability
// that one or more quanta of neurotransmitter are released when the
// stimulus is presented) averaged 0.39 (range, 0.20-0.59), and the
// success probability for the second pulse (5-100 ms interpulse
// interval) averaged 0.89 (range, 0.68-0.95). For these neurons, the
// success probability for a second stimulus appeared to be
// independent of the probability that the first stimulus caused
// release (Figure 1A).  The data in Figure 1 are restricted to the
// situation in which no release occured to the first stimulus or, if
// such release did occur, for interstimulus times of greater than
// about 20 ms."


// 

// Change log:
// 
// 2010-12-13 15:56:23 (+0530) Initial version by Subhasis Ray
// 

// Code:

#ifndef   	STOCHSPIKEGEN_H_
#define   	STOCHSPIKEGEN_H_

#include "SpikeGen.h"

class StochSpikeGen: SpikeGen
{
  public:
    StochSpikeGen(){
        Pr_ = 1.0;
    }
    static void setPr(const Conn* conn, double value);
    static double getPr(Eref e);

    void innerProcessFunc(const Conn *c, ProcInfo p);
  protected:
    double Pr_;
};

#endif 	    /* !STOCHSPIKEGEN_H_ */




// 
// StochSpikeGen.h ends here
