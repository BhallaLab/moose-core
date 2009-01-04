// RC.cpp --- 
// 
// Filename: RC.cpp
// Description: 
// Author: subhasis ray
// Maintainer: 
// Created: Wed Dec 31 15:47:45 2008 (+0530)
// Version: 
// Last-Updated: Sat Jan  3 21:44:07 2009 (+0530)
//           By: subhasis ray
//     Update #: 98
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
                        "Output value of the RC circuit." ),
        new ValueFinfo( "inject", ValueFtype1< double >::global(),
                        GFCAST( &RC::getInject ),
                        RFCAST( &RC::setInject ),
                        "Input value to the RC circuit." ),
        process,
        new SrcFinfo( "outputSrc", Ftype1< double >::global(),
                      "Sends the output of the RC circuit." ),
        new DestFinfo( "injectDest", Ftype1< double >::global(),
                       RFCAST( &RC::setInject ),
                       "Receives input to the RC circuit." ),
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
        isteps_(1)
{
    // Do nothing
}
            
void RC::setV0( const Conn& conn, double v0 )
{
    RC* instance = static_cast< RC* >( conn.data() );
    instance->v0_ = v0;
}

double RC::getV0( Eref e )
{
    RC* instance = static_cast< RC* >( e.e->data() );
    return instance->v0_;
}

void RC::setResistance( const Conn& conn, double resistance )
{
    RC* instance = static_cast< RC* >( conn.data() );
    instance->resistance_ = resistance;
}

double RC::getResistance( Eref e )
{
    RC* instance = static_cast< RC* >( e.e->data() );
    return instance->resistance_;
}

void RC::setCapacitance( const Conn& conn, double capacitance )
{
    RC* instance = static_cast< RC* >( conn.data());
    instance->capacitance_ = capacitance;
}

double RC::getCapacitance( Eref e )
{
    RC* instance = static_cast< RC* >( e.e->data() );
    return instance->capacitance_;
}

double RC::getState( Eref e )
{
    RC* instance = static_cast< RC* >( e.e->data() );
    return instance->state_;
}

void RC::setInject( const Conn& conn, double inject )
{
    RC* instance = static_cast< RC* >( conn.data() );
    instance->inject_ = inject;
}

double RC::getInject( Eref e )
{
    RC* instance = static_cast< RC* >( e.e->data() );
    return instance->inject_;
}

void RC::processFunc( const Conn& conn, ProcInfo proc )
{
    RC* instance = static_cast< RC* >( conn.data() );
    static double inject_prev = instance->inject_;
    double dVin = (instance->inject_ - inject_prev) * instance->resistance_;
    double Vin = instance->inject_ * instance->resistance_;
    instance->state_ =
            (instance->state_ -
             Vin +
             instance->resistance_ * instance->capacitance_ * dVin / instance->dt_) *
            instance->exp_ +
            ( Vin +
              dVin +
              instance->resistance_ * instance->capacitance_* dVin / instance->dt_ );
            
    //    double k1, k2, k3, k4; // Terms for 4-th order Runge-Kutta
    // for ( int i = 0; i < instance->isteps_; ++i ){
    //     double i_total = instance->inject_ - instance->state_ / instance->resistance_;
    //     k1 = dt * i_total / instance->capacitance_;
    //     k2 = dt * ( instance->state_ + k1 / 2.0 );
    //     k3 = dt * ( instance->state_ + k2 / 2.0 );
    //     k4 = dt * ( instance->state_ + k3 );
    //     instance->state_ += (k1 + 2 * k2 + 2 * k3 + k4) / 6.0;
    // }
    inject_prev = instance->inject_;
}

void RC::reinitFunc( const Conn& conn, ProcInfo proc)
{
    RC* instance = static_cast< RC* >(conn.data());
    double tau = instance->resistance_ * instance->capacitance_;
    if ( tau >= 20 * proc->dt_) {
        instance->isteps_ = 1;
    } else { // take care of dt that is large compared to tau
        instance->isteps_ = (int)(20 * proc->dt_ / tau);
    }
    // not sure if reinit and procss use the same ProcInfo
    instance->dt_ = proc->dt_ / instance->isteps_;
    instance->state_ = instance->v0_;
    instance->exp_ = exp(-instance->dt_ / tau);
}


// 
// RC.cpp ends here
