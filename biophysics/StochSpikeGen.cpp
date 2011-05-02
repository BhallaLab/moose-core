// StochSpikeGen.cpp --- 
// 
// Filename: StochSpikeGen.cpp
// Description: 
// Author: Subhasis Ray
// Maintainer: 
// Copyright (C) 2010 Subhasis Ray, all rights reserved.
// Created: Thu Apr 14 17:48:14 2011 (+0530)
// Version: 
// Last-Updated: Mon May  2 11:31:33 2011 (+0530)
//           By: Subhasis Ray
//     Update #: 88
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
                new ValueFinfo( "pr", ValueFtype1< double >::global(),
                                GFCAST( &StochSpikeGen::getPr ),
                                RFCAST( &StochSpikeGen::setPr ),
                                "Release probability."
                                ),
            };

    static string doc[] =
            {
		"Name", "StochSpikeGen",
		"Author", "Subhasis Ray",
		"Description", "StochSpikeGen object, is a generalization of SpikeGen object "
                "where it transmits a spike with a release probability specified by "
                "Pr (=0.95 by default)."
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

double StochSpikeGen::getPr(Eref e)
{
    return static_cast<StochSpikeGen*>(e.data())->Pr_;
}

void StochSpikeGen::setPr(const Conn * conn, double value)
{
    static_cast< StochSpikeGen* >(conn->data())->Pr_ = value;
}

void StochSpikeGen::innerProcessFunc(const Conn* conn, ProcInfo p)
{
    double t = p->currTime_;
    if ( V_ > threshold_ ) {
        if ((t + p->dt_/2.0) >= (lastEvent_ + refractT_)){
            if (!edgeTriggered_ || (edgeTriggered_ && !fired_)) {
                if (mtrand() > Pr_){
                    send1< double >( conn->target(), eventSlot, t );
                    lastEvent_ = t;
                    state_ = amplitude_;
                    fired_ = true;
                }
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
