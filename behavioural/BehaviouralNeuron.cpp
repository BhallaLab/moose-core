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
    vals_["tau"] = 0.0;
    vals_["gL"] = 1.0;
    vals_["t"] = currTime_;

    // Following are inherited from Compartment.
    vals_["Vm"] = Vm_;
    vals_["Cm"] = Cm_;
    vals_["v"] = Vm_;
    vals_["EL"] = 0.0;
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
    VmOut()->send( e, Vm_ );

    // Eval parser.
    vector<double> dy;
    double v = 0.0;
    for( auto p : odeMap_ )
    {
        try
        {
            v = p.second.Eval();
        }
        catch( mu::ParserError& e )  
        {
            cout << "Error in evaluation: " << e.GetMsg() << endl;
        }
        dy.push_back(v);
    }
    for( auto y : dy )
        cout << y << ',';
}

void BehaviouralNeuron::vReinit(  const Eref& e, ProcPtr p )
{
    activation_ = 0.0;
    fired_ = false;
    lastEvent_ = -refractT_; // Allow it to fire right away.
    if(! isBuilt_ )
        buildSystem( );
}


void BehaviouralNeuron::setupParser(mu::Parser& p)
{
    LOG( moose::debug, "Setting up parser" );
    for( auto v : vals_ )
        p.DefineVar( v.first, &v.second );

}

void BehaviouralNeuron::buildSystem( )
{
    for( auto eq : eqs_ )
    {
        size_t loc = eq.find( '=' );
        if( loc == std::string::npos)
        {
            LOG( moose::warning, "Invalid equation: " << eq << ". Ignored!" );
            continue;
        }

        auto lhs = moose::trim(eq.substr(0, loc));
        auto rhs = moose::trim(eq.substr(loc+1));

        mu::Parser p;
        setupParser(p);
        p.SetExpr(rhs);
        odeMap_[lhs] = p;
    }

    isBuilt_ = true;
}
