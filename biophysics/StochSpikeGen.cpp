// StochSpikeGen.cpp --- 
// 
// Filename: StochSpikeGen.cpp
// Description: 
// Author: Subhasis Ray
// Maintainer: 
// Copyright (C) 2010 Subhasis Ray, all rights reserved.
// Created: Thu Apr 14 17:48:14 2011 (+0530)
// Version: 
// Last-Updated: Fri Apr 15 16:12:04 2011 (+0530)
//           By: Subhasis Ray
//     Update #: 81
// URL: 
// Keywords: 
// Compatibility: 
// 
// 

// Commentary: 
// 
// A spike generator with option for failure probability.
// 
// 

// Change log:
// 
// 
// 

// Code:

#include "moose.h"

#include "StochSpikeGen.h"

extern double mtrand(void);

const Cinfo* initStochSpikeGenCinfo()
{
    static Finfo* stochSpikeGenFinfos[] = 
            {
                new ValueFinfo( "failureP", ValueFtype1< double >::global(),
                                GFCAST( &StochSpikeGen::getFailureP ),
                                RFCAST( &StochSpikeGen::setFailureP ),
                                "Failure probability of the spikegen object."
                                ),
            };

    static string doc[] =
            {
		"Name", "StochSpikeGen",
		"Author", "Subhasis Ray",
		"Description", "StochSpikeGen object, is a generalization of SpikeGen object "
                "where it transmits a spike with a failure probability specified by "
                "failureP (=0.05 by default)."
            };
    static Cinfo stochSpikeGenCinfo(
            doc,
            sizeof( doc ) / sizeof( string ),
            initSpikeGenCinfo(),
            stochSpikeGenFinfos,
            sizeof( stochSpikeGenFinfos ) / sizeof( Finfo* ),
            ValueFtype1< StochSpikeGen >::global());

    return &stochSpikeGenCinfo;
}

static const Cinfo* stochSpikeGenCinfo = initStochSpikeGenCinfo();

static const Slot eventSlot =
	initStochSpikeGenCinfo()->getSlot( "event" );

double StochSpikeGen::getFailureP(Eref e)
{
    return static_cast<StochSpikeGen*>(e.data())->failureP_;
}

void StochSpikeGen::setFailureP(const Conn * conn, double value)
{
    static_cast< StochSpikeGen* >(conn->data())->failureP_ = value;
}

void StochSpikeGen::innerProcessFunc(const Conn* conn, ProcInfo p)
{
    cout << "## Here" << endl;
    double t = p->currTime_;
    if ( V_ > threshold_ ) {
        if ((t + p->dt_/2.0) >= (lastEvent_ + refractT_)){
            if (!edgeTriggered_ || (edgeTriggered_ && !fired_)) {
                if (mtrand() < failureP_){
                    return;
                }
                send1< double >( conn->target(), eventSlot, t );
                lastEvent_ = t;
                state_ = amplitude_;
                fired_ = true;                    
            } else {
                state_ = 0.0;                
            }
        }
    } else {
        state_ = 0.0;
        fired_ = false;
    }    
}


// 
// StochSpikeGen.cpp ends here
