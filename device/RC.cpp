// RC.cpp --- 
// 
// Filename: RC.cpp
// Description: 
// Author: subhasis ray
// Maintainer: 
// Created: Wed Dec 31 15:47:45 2008 (+0530)
// Version: 
// Last-Updated: Fri Jun  4 00:03:11 2010 (+0530)
//           By: Subhasis Ray
//     Update #: 167
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

#include "RC.h"

const Cinfo* initRCCinfo()
{
    static Finfo* processShared[] = {
        new DestFinfo( "process", Ftype1< ProcInfo >::global(),
                       RFCAST( &RC::processFunc)),
        new DestFinfo( "reinit", Ftype1< ProcInfo >::global(),
                       RFCAST( &RC::reinitFunc )),
    };
    static Finfo* process = new SharedFinfo( "process", processShared, sizeof( processShared ) / sizeof( Finfo* ));
    static Finfo* rcFinfos[] = {
        new ValueFinfo( "V0", ValueFtype1< double >::global(),
                        GFCAST( &RC::getV0 ),
                        RFCAST( &RC::setV0 ),
                        "Initial value of 'state'" ),
        new ValueFinfo( "R", ValueFtype1< double >::global(),
                        GFCAST( &RC::getResistance ),
                        RFCAST( &RC::setResistance ),
                        "Series resistance of the RC circuit." ),
        new ValueFinfo( "C", ValueFtype1< double >::global(),
                        GFCAST( &RC::getCapacitance ),
                        RFCAST( &RC::setCapacitance ),
                        "Parallel capacitance of the RC circuit." ),
        new ValueFinfo( "state", ValueFtype1< double >::global(),
                        GFCAST( &RC::getState ),
                        RFCAST( &dummyFunc ),
                        "Output value of the RC circuit. This is the voltage across the capacitor." ),
        new ValueFinfo( "inject", ValueFtype1< double >::global(),
                        GFCAST( &RC::getInject ),
                        RFCAST( &RC::setInject ),
                        "Input value to the RC circuit.This is handled as an input current to the circuit." ),
        process,
        new SrcFinfo( "outputSrc", Ftype1< double >::global(),
                      "Sends the output of the RC circuit." ),
        new DestFinfo( "injectMsg", Ftype1< double >::global(),
                       RFCAST( &RC::setInjectMsg ),
                       "Receives input to the RC circuit. All incoming messages are summed up to give the total input current." ),
    };
    static SchedInfo  schedInfo[] = {{process, 0, 0}};
    static string doc[] = {
        "Name", "RC",
        "Author", "Subhasis Ray, 2008, NCBS",
        "Description", "RC circuit: a series resistance R shunted by a capacitance C." };
    static Cinfo rcCinfo(
            doc,
            sizeof( doc ) / sizeof( string ),
            initNeutralCinfo(),
            rcFinfos,
            sizeof( rcFinfos ) / sizeof( Finfo* ),
            ValueFtype1< RC >::global(),
            schedInfo, 1 );
    return &rcCinfo;
}

static const Cinfo* rcCinfo = initRCCinfo();

static const Slot outputSlot = initRCCinfo()->getSlot( "outputSrc" );

RC::RC():
        v0_(0),
        resistance_(1.0),
        capacitance_(1.0),
        state_(0),
        inject_(0),
        msg_inject_(0.0),
        exp_(0.0),
        dt_tau_(0.0)
{
    // Do nothing
}
            
void RC::setV0( const Conn* conn, double v0 )
{
    RC* instance = static_cast< RC* >( conn->data() );
    instance->v0_ = v0;
}

double RC::getV0( Eref e )
{
    RC* instance = static_cast< RC* >( e.data() );
    return instance->v0_;
}

void RC::setResistance( const Conn* conn, double resistance )
{
    RC* instance = static_cast< RC* >( conn->data() );
    instance->resistance_ = resistance;
}

double RC::getResistance( Eref e )
{
    RC* instance = static_cast< RC* >( e.data() );
    return instance->resistance_;
}

void RC::setCapacitance( const Conn* conn, double capacitance )
{
    RC* instance = static_cast< RC* >( conn->data());
    instance->capacitance_ = capacitance;
}

double RC::getCapacitance( Eref e )
{
    RC* instance = static_cast< RC* >( e.data() );
    return instance->capacitance_;
}

double RC::getState( Eref e )
{
    RC* instance = static_cast< RC* >( e.e->data() );
    return instance->state_;
}

void RC::setInject( const Conn* conn, double inject )
{
    RC* instance = static_cast< RC* >( conn->data() );
    instance->inject_ = inject;
}

double RC::getInject( Eref e )
{
    RC* instance = static_cast< RC* >( e.e->data() );
    return instance->inject_;
}

void RC::setInjectMsg( const Conn* conn, double inject )
{
    RC* instance = static_cast< RC* >( conn->data() );
    instance->msg_inject_ += inject;
}

/**
   calculates the new voltage across the capacitor.  this is the exact
   solution as described in Electronic Circuit and System Simulation
   Methods by Lawrance Pillage, McGraw-Hill Professional, 1999. pp
   87-100. Eqn: 4.7.21 */

void RC::processFunc( const Conn* conn, ProcInfo proc )
{
    RC* instance = static_cast< RC* >( conn->data() );
    double sum_inject_prev = instance->inject_ + instance->msg_inject_;
    double sum_inject = instance->inject_ + instance->msg_inject_;
    double dVin = (sum_inject - sum_inject_prev) * instance->resistance_;
    double Vin = sum_inject * instance->resistance_;
    instance->state_ = Vin + dVin - dVin / instance->dt_tau_ +
            (instance->state_ - Vin + dVin / instance->dt_tau_) * instance->exp_;
    sum_inject_prev = sum_inject;
    instance->msg_inject_ = 0.0;
    send1<double>(conn->target(), outputSlot, instance->state_);
}

void RC::reinitFunc( const Conn* conn, ProcInfo proc)
{
    RC* instance = static_cast< RC* >(conn->data());
    instance->dt_tau_ = proc->dt_ / (instance->resistance_ * instance->capacitance_);
    instance->state_ = instance->v0_;
    if (instance->dt_tau_ > 1e-15){ 
        instance->exp_ = exp(-instance->dt_tau_);
    } else {// use approximation
        instance->exp_ = 1 - instance->dt_tau_;
    }
    instance->msg_inject_ = 0.0;
}


// 
// RC.cpp ends here
