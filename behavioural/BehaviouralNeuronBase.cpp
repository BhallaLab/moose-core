/***
 *    Description:  Behavioural Neuron.
 *
 *        Created:  2018-12-28

 *         Author:  Dilawar Singh <dilawars@ncbs.res.in>
 *   Organization:  NCBS Bangalore
 *        License:  See the LICENSE file.
 */

#include <memory>

#include "../basecode/header.h"
#include "../basecode/ElementValueFinfo.h"
#include "../biophysics/CompartmentBase.h"
#include "../biophysics/Compartment.h"
#include "BehaviouralNeuronBase.h"
#include "BehaviouralSystem.h"

#include "../utility/print_function.hpp"
#include "../utility/strutil.h"

using namespace moose;
using namespace std;

/* This Finfo is used to send out Vm and spike to other elements. */
SrcFinfo1< double >* BehaviouralNeuronBase::spikeOut()
{
    static SrcFinfo1< double > spikeOut(
        "spikeOut",
        "Sends out spike events. The argument is the timestamp of "
        "the spike. "
    );
    return &spikeOut;
}

const Cinfo* BehaviouralNeuronBase::initCinfo()
{
    //////////////////////////////////////////////////////////////
    // Field Definitions
    //////////////////////////////////////////////////////////////
    static ElementValueFinfo< BehaviouralNeuronBase, double > thresh(
        "thresh",
        "firing threshold",
        &BehaviouralNeuronBase::setThresh,
        &BehaviouralNeuronBase::getThresh
    );

    static ElementValueFinfo< BehaviouralNeuronBase, vector<string> > variables(
        "variables",
        "ODE variables.",
        &BehaviouralNeuronBase::setODEVariables,
        &BehaviouralNeuronBase::getODEVariables
    );

    static ElementValueFinfo< BehaviouralNeuronBase, vector<string> > states(
        "states",
        "ODE states.",
        &BehaviouralNeuronBase::setODEStates,
        &BehaviouralNeuronBase::getODEStates
    );

    static ElementValueFinfo<BehaviouralNeuronBase, vector<string>> equations(
        "equations",
        "Set equations.",
        &BehaviouralNeuronBase::setEquations,
        &BehaviouralNeuronBase::getEquations
    );

    static ElementValueFinfo< BehaviouralNeuronBase, double > vReset(
        "vReset",
        "voltage is set to vReset after firing",
        &BehaviouralNeuronBase::setVReset,
        &BehaviouralNeuronBase::getVReset
    );

    static ElementValueFinfo< BehaviouralNeuronBase, double > refractoryPeriod(
        "refractoryPeriod",
        "Minimum time between successive spikes",
        &BehaviouralNeuronBase::setRefractoryPeriod,
        &BehaviouralNeuronBase::getRefractoryPeriod
    );

    static ReadOnlyElementValueFinfo< BehaviouralNeuronBase, double > lastEventTime(
        "lastEventTime",
        "Timestamp of last firing.",
        &BehaviouralNeuronBase::getLastEventTime
    );

    static ReadOnlyElementValueFinfo< BehaviouralNeuronBase, bool > hasFired(
        "hasFired",
        "The object has fired within the last timestep",
        &BehaviouralNeuronBase::hasFired
    );


    //////////////////////////////////////////////////////////////
    // MsgDest Definitions
    //////////////////////////////////////////////////////////////
    static DestFinfo activation(
        "activation",
        "Handles value of synaptic activation arriving on this object",
        new OpFunc1< BehaviouralNeuronBase, double >( &BehaviouralNeuronBase::activation ));

    //////////////////////////////////////////////////////////////

    static Finfo* behaviouralNeuronFinfos[] =
    {
        &thresh,                            // Value
        &variables,                         // Value
        &states,                            // Value
        &equations,                         // Value
        &vReset,                            // Value
        &refractoryPeriod,                  // Value
        &hasFired,                          // ReadOnlyValue
        &lastEventTime,                     // ReadOnlyValue
        &activation,                        // DestFinfo
        BehaviouralNeuronBase::spikeOut(),  // MsgSrc
    };

    static string doc[] =
    {
        "Name", "BehaviouralNeuronBase",
        "Author", "Dilawar Singh",
        "Description", "Base class for Behavioural Neurons.",
    };
    static ZeroSizeDinfo< int > dinfo;
    static Cinfo behaviouralNeuronBaseCinfo(
        "BehaviouralNeuronBase",
        Compartment::initCinfo(),
        behaviouralNeuronFinfos,
        sizeof( behaviouralNeuronFinfos ) / sizeof (Finfo*),
        &dinfo,
        doc,
        sizeof(doc)/sizeof(string)
    );

    return &behaviouralNeuronBaseCinfo;
}

static const Cinfo* behaviouralNeuronBaseCinfo = BehaviouralNeuronBase::initCinfo();

BehaviouralNeuronBase::BehaviouralNeuronBase() :
    threshold_( 0.0 ),
    vReset_( 0.0 ),
    activation_( 0.0 ),
    refractT_( 0.0 ),
    lastEvent_( 0.0 ),
    fired_( false )
{
}
    
BehaviouralNeuronBase::~BehaviouralNeuronBase()
{
}

// Value Field access function definitions.
void BehaviouralNeuronBase::setThresh( const Eref& e, double val )
{
    threshold_ = val;
}

double BehaviouralNeuronBase::getThresh( const Eref& e ) const
{
    return threshold_;
}

void BehaviouralNeuronBase::setVReset( const Eref& e, double val )
{
    vReset_ = val;
}

double BehaviouralNeuronBase::getVReset( const Eref& e ) const
{
    return vReset_;
}

void BehaviouralNeuronBase::setRefractoryPeriod( const Eref& e, double val )
{
    refractT_ = val;
}

double BehaviouralNeuronBase::getRefractoryPeriod( const Eref& e ) const
{
    return refractT_;
}

double BehaviouralNeuronBase::getLastEventTime( const Eref& e ) const
{
    return lastEvent_;
}

bool BehaviouralNeuronBase::hasFired( const Eref& e ) const
{
    return fired_;
}

//////////////////////////////////////////////////////////////////
// BehaviouralNeuronBase::Dest function definitions.
//////////////////////////////////////////////////////////////////
void BehaviouralNeuronBase::activation( double v )
{
    activation_ += v;
}

// Variables
void BehaviouralNeuronBase::setODEVariables( const Eref& e, const vector<string> vars)
{
    variables_ = vars;
}

vector<string> BehaviouralNeuronBase::getODEVariables( const Eref& e) const
{
    return variables_;
}

// States
void BehaviouralNeuronBase::setODEStates( const Eref& e, const vector<string> ss)
{
    states_ = ss;
}

vector<string> BehaviouralNeuronBase::getODEStates( const Eref& e) const
{
    return states_;
}

// Equations.
void BehaviouralNeuronBase::setEquations( const Eref& e, const vector<string> eqs)
{
    eqs_ = eqs;
}

vector<string> BehaviouralNeuronBase::getEquations( const Eref& e) const
{
    return eqs_;
}


