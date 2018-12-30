/***
 *    Description:  Behavioural neuron.
 *
 *        Created:  2018-12-28

 *         Author:  Dilawar Singh <dilawars@ncbs.res.in>
 *   Organization:  NCBS Bangalore
 *        License:  See LICENSE file.
 */

#include "../basecode/header.h"
#include "../basecode/ElementValueFinfo.h"
#include "../biophysics/CompartmentBase.h"
#include "../biophysics/Compartment.h"
#include "BehaviouralNeuronBase.h"
#include "BehaviouralNeuron.h"
#include "OdeSystem.h"
#include "../external/muparser/include/muParser.h"
#include "../utility/strutil.h"
#include "../utility/print_function.hpp"

using namespace moose;

const Cinfo* BehaviouralNeuron::initCinfo()
{
    static string doc[] =
    {
        "Name", "BehaviouralNeuron",
        "Author", "Dilawar Singh",
        "Description", "A neuron whose behaviour is described by ODEs."
    };

    static Dinfo< BehaviouralNeuron > dinfo;
    static Cinfo lifCinfo(
        "BehaviouralNeuron",
        BehaviouralNeuronBase::initCinfo(),
        0, 0,
        &dinfo,
        doc,
        sizeof(doc)/sizeof(string)
    );
    return &lifCinfo;
}

static const Cinfo* lifCinfo = BehaviouralNeuron::initCinfo();

//////////////////////////////////////////////////////////////////
// Here we put the Compartment class functions.
//////////////////////////////////////////////////////////////////

BehaviouralNeuron::BehaviouralNeuron()
{
    vals_["t"]  = &currTime_;

    // Following are inherited from Compartment.
    vals_["Vm"] = &Vm_;
    vals_["Cm"] = &Cm_;
    vals_["v"]  = &Vm_;
}

BehaviouralNeuron::~BehaviouralNeuron()
{
    ;
}

//////////////////////////////////////////////////////////////////
// BehaviouralNeuron::Dest function definitions.
//////////////////////////////////////////////////////////////////
void BehaviouralNeuron::vProcess( const Eref& e, ProcPtr p )
{
    fired_ = false;
    // For parser.
    currTime_ = p->currTime;

    if ( p->currTime < lastEvent_ + refractT_ )
    {
        Vm_ = vReset_;
        sumInject_ = 0.0;
    }
    else
    {
        // activation can be a continous variable (graded synapse).
        // So integrate it at every time step, thus *dt.
        // For a delta-fn synapse, SynHandler-s divide by dt and send activation.
        // See: http://www.genesis-sim.org/GENESIS/Hyperdoc/Manual-26.html#synchan
        //          for this continuous definition of activation.
        Vm_ += activation_ * p->dt;
        activation_ = 0.0;
        if ( Vm_ > threshold_ )
        {
            Vm_ = vReset_;
            lastEvent_ = p->currTime;
            fired_ = true;
            spikeOut()->send( e, p->currTime );
        }
        else
            Compartment::vProcess(e, p);
    }

    // Eval parser.
    double v = 0.0;
    string var;
    for( auto p : odeMap_ )
    {
        try {
            v = p.second.Eval();
        } catch(mu::ParserError& e){
            cout << "Error in evaluation: " << e.GetMsg() << endl;
        }

        // p.first is usually x', y' or v'. When updating the variable value,
        // remove ' from the map.
        var = p.first; var.pop_back();
        *(vals_[var]) += v;
    }
    VmOut()->send( e, Vm_ );
}

void BehaviouralNeuron::vReinit(  const Eref& e, ProcPtr p )
{
    activation_ = 0.0;
    fired_ = false;
    lastEvent_ = -refractT_; // Allow it to fire right away.
    if(! isBuilt_ )
        buildSystem( );
}


