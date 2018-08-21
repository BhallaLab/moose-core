/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "../basecode/header.h"
#include "../basecode/global.h"

#include "RandSpike.h"

///////////////////////////////////////////////////////
// MsgSrc definitions
///////////////////////////////////////////////////////
static SrcFinfo1< double > *spikeOut()
{
    static SrcFinfo1< double > spikeOut( "spikeOut",
                                         "Sends out a trigger for an event.");
    return &spikeOut;
}

const Cinfo* RandSpike::initCinfo()
{
    ///////////////////////////////////////////////////////
    // Shared message definitions
    ///////////////////////////////////////////////////////
    static DestFinfo process( "process",
                              "Handles process call",
                              new ProcOpFunc< RandSpike >( &RandSpike::process ) );
    static DestFinfo reinit( "reinit",
                             "Handles reinit call",
                             new ProcOpFunc< RandSpike >( &RandSpike::reinit ) );

    static Finfo* processShared[] =
    {
        &process, &reinit
    };

    static SharedFinfo proc( "proc",
                             "Shared message to receive Process message from scheduler",
                             processShared, sizeof( processShared ) / sizeof( Finfo* ) );

    //////////////////////////////////////////////////////////////////
    // Dest Finfos.
    //////////////////////////////////////////////////////////////////

    //////////////////////////////////////////////////////////////////
    // Value Finfos.
    //////////////////////////////////////////////////////////////////

    static ValueFinfo< RandSpike, double > rate( "rate",
            "Specifies rate for random spike train. Note that this is"
            "probabilistic, so the instantaneous rate may differ. "
            "If the rate is assigned be message and it varies slowly then "
            "the average firing rate will approach the specified rate",
            &RandSpike::setRate,
            &RandSpike::getRate
                                               );
    static ValueFinfo< RandSpike, double > refractT( "refractT",
            "Refractory Time.",
            &RandSpike::setRefractT,
            &RandSpike::getRefractT
                                                   );
    static ValueFinfo< RandSpike, double > absRefract( "abs_refract",
            "Absolute refractory time. Synonym for refractT.",
            &RandSpike::setRefractT,
            &RandSpike::getRefractT
                                                     );
    static ValueFinfo< RandSpike, bool > doPeriodic( "doPeriodic",
            "Flag: when false, do Poisson process with specified mean rate.\n"
            "When true, fire periodically at specified rate.\n"
            "Defaults to false. Note that refractory time overrides this: "
            "Rate cannot exceed 1/refractT.",
            &RandSpike::setDoPeriodic,
            &RandSpike::getDoPeriodic
                                                   );
    static ReadOnlyValueFinfo< RandSpike, bool > hasFired( "hasFired",
            "True if RandSpike has just fired",
            &RandSpike::getFired
                                                         );

    static Finfo* spikeGenFinfos[] =
    {
        spikeOut(),	// SrcFinfo
        &proc,		// Shared
        &rate,		// Value
        &refractT,	// Value
        &absRefract,	// Value
        &doPeriodic,	// Value
        &hasFired,	// ReadOnlyValue
    };

    static string doc[] =
    {
        "Name", "RandSpike",
        "Author", "Upi Bhalla",
        "Description", "RandSpike object, generates random or regular "
        "spikes at "
        "specified mean rate. Based closely on GENESIS randspike. "
    };
    static Dinfo< RandSpike > dinfo;
    static Cinfo spikeGenCinfo(
        "RandSpike",
        Neutral::initCinfo(),
        spikeGenFinfos, sizeof( spikeGenFinfos ) / sizeof( Finfo* ),
        &dinfo,
        doc,
        sizeof(doc)/sizeof(string)
    );

    return &spikeGenCinfo;
}

static const Cinfo* spikeGenCinfo = RandSpike::initCinfo();

RandSpike::RandSpike()
    :
    rate_( 0.0 ),
    realRate_( 0.0 ),
    refractT_(0.0),
    lastEvent_(0.0),
    threshold_(0.0),
    fired_( false ),
    doPeriodic_( false )
{
    ;
}

//////////////////////////////////////////////////////////////////
// Here we put the RandSpike class functions.
//////////////////////////////////////////////////////////////////

// Value Field access function definitions.
void RandSpike::setRate( double rate )
{
    if ( rate < 0.0 )
    {
        cout <<"Warning: RandSpike::setRate: Rate must be >= 0. Using 0.\n";
        rate = 0.0;
    }
    rate_ = rate;
    double prob = 1.0 - rate * refractT_;
    if ( prob <= 0.0 )
    {
        cout << "Warning: RandSpike::setRate: Rate is too high compared to refractory time\n";
        realRate_ = rate_;
    }
    else
    {
        realRate_ = rate_ / prob;
    }
}
double RandSpike::getRate() const
{
    return rate_;
}

void RandSpike::setRefractT( double val )
{
    refractT_ = val;
}
double RandSpike::getRefractT() const
{
    return refractT_;
}

bool RandSpike::getFired() const
{
    return fired_;
}

void RandSpike::setDoPeriodic( bool val )
{
    doPeriodic_ = val;
}
bool RandSpike::getDoPeriodic() const
{
    return doPeriodic_;
}


//////////////////////////////////////////////////////////////////
// RandSpike::Dest function definitions.
//////////////////////////////////////////////////////////////////

void RandSpike::process( const Eref& e, ProcPtr p )
{
    if ( refractT_ > p->currTime - lastEvent_  || rate_ <= 0.0 )
        return;

    fired_ = false;
    if (doPeriodic_)
    {
        if ( (p->currTime - lastEvent_) > 1.0/rate_ )
        {
            lastEvent_ = p->currTime;
            spikeOut()->send( e, p->currTime );
            fired_ = true;
        }
    }
    else
    {
        double prob = realRate_ * p->dt;
        if ( prob >= 1.0 || prob >= moose::mtrand() )
        {
            lastEvent_ = p->currTime;
            spikeOut()->send( e, p->currTime );
            fired_ = true;
        }
    }
}

// Set it so that first spike is allowed.
void RandSpike::reinit( const Eref& e, ProcPtr p )
{
    if ( rate_ <= 0.0 )
    {
        lastEvent_ = 0.0;
        realRate_ = 0.0;
    }
    else
    {
        double prob = moose::mtrand();
        double m = 1.0 / rate_;
        lastEvent_ = m * log( prob );
    }
}
