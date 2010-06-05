// PIDController.cpp --- 
// 
// Filename: PIDController.cpp
// Description: 
// Author: subhasis ray
// Maintainer: 
// Created: Tue Dec 30 23:36:01 2008 (+0530)
// Version: 
// Last-Updated: Fri Jun  4 14:35:26 2010 (+0530)
//           By: Subhasis Ray
//     Update #: 222
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

#include "PIDController.h"

const Cinfo* initPIDControllerCinfo()
{
    static Finfo* processShared[] = {
        new DestFinfo( "process", Ftype1< ProcInfo>::global(),
		      RFCAST( &PIDController::processFunc )),
	new DestFinfo( "reinit", Ftype1< ProcInfo >::global(),
                       RFCAST( &PIDController::reinitFunc )),
    };
    static Finfo* process = new SharedFinfo( "process", processShared, sizeof( processShared ) / sizeof( Finfo* ));
    
    static Finfo* pidFinfos[] = {
        new ValueFinfo( "gain", ValueFtype1< double >::global(),
                        GFCAST( &PIDController::getGain ),
                        RFCAST( &PIDController::setGain ),
                        "This is the proportional gain (Kp). This tuning parameter scales the"
                        " proportional term. Larger gain usually results in faster response, but"
                        " too much will lead to instability and oscillation." ),
        new ValueFinfo( "saturation",  ValueFtype1< double >::global(),
                         GFCAST( &PIDController::getSaturation ),
                         RFCAST( &PIDController::setSaturation ),
                        "Bound on the permissible range of output. Defaults to maximum double"
                        " value." ),
        new ValueFinfo( "command", ValueFtype1< double >::global(),
                        GFCAST( &PIDController::getCommand ),
                        RFCAST( &PIDController::setCommand  ),
                        "The command (desired) value of the sensed parameter. In control theory"
                        " this is commonly known as setpoint(SP)." ),
        new ValueFinfo( "sensed", ValueFtype1< double >::global(),
                         GFCAST( &PIDController::getSensed ),
                         RFCAST( &dummyFunc ),
                        "Sensed (measured) value. This is commonly known as process variable"
                        "(PV) in control theory."),
        new ValueFinfo( "tauI", ValueFtype1< double >::global(),
                        GFCAST( &PIDController::getTauI ),
                        RFCAST( &PIDController::setTauI ),
                        "The integration time constant, typically = dt. This is actually"
                        " proportional gain divided by integral gain (Kp/Ki)). Larger Ki"
                        " (smaller tauI) usually leads to fast elimination of steady state"
                        " errors at the cost of larger overshoot." ),
        new ValueFinfo( "tauD", ValueFtype1< double >::global(),
                         GFCAST( &PIDController::getTauD ),
                         RFCAST( &PIDController::setTauD ),
                        "The differentiation time constant, typically = dt / 4. This is"
                        " derivative gain (Kd) times proportional gain (Kp). Larger Kd (tauD)"
                        " decreases overshoot at the cost of slowing down transient response"
                        " and may lead to instability."),
        new ValueFinfo( "output", ValueFtype1< double >::global(),
                         GFCAST( &PIDController::getOutput ),
                         RFCAST( &dummyFunc ),
                         "Output of the PIDController. This is given by:"
                         "      gain * ( error + INTEGRAL[ error dt ] / tau_i   + tau_d * d(error)/dt )\n"
                         "Where gain = proportional gain (Kp), tau_i = integral gain (Kp/Ki) and"
                         " tau_d = derivative gain (Kd/Kp). In control theory this is also known"
                         " as the manipulated variable (MV)"),
        new ValueFinfo( "error", ValueFtype1< double >::global(),
                        GFCAST( &PIDController::getError ),
                        RFCAST( &dummyFunc ),
                        "The error term, which is the difference between command and sensed"
                        " value."),
        new ValueFinfo( "integral",ValueFtype1< double >::global(),
                        GFCAST( &PIDController::getEIntegral ),
                        RFCAST( &dummyFunc ),
                        "The integral term. It is calculated as INTEGRAL(error dt) ="
                        " previous_integral + dt * (error + e_previous)/2."),
        new ValueFinfo( "derivative",ValueFtype1< double >::global(),
                        GFCAST( &PIDController::getEDerivative ),
                        RFCAST( &dummyFunc ),
                        "The derivative term. This is (error - e_previous)/dt."),
        new ValueFinfo( "e_previous",ValueFtype1< double >::global(),
                        GFCAST( &PIDController::getEPrevious ),
                        RFCAST( &dummyFunc ),
                        "The error term for previous step."),
        process,
        new SrcFinfo( "outputSrc", Ftype1< double >::global(),
                      "Sends the output of the PIDController. This is known as manipulated"
                      " variable (MV) in control theory. This should be fed into the process"
                      " which we are trying to control." ),
        new DestFinfo( "commandDest", Ftype1< double >::global(),
                       RFCAST( &PIDController::setCommand ),
                       "Command (desired value) input. This is known as setpoint (SP) in"
                       " control theory." ),
        new DestFinfo( "sensedDest", Ftype1< double >::global(),
                       RFCAST( &PIDController::setSensed ),
                       "Sensed parameter - this is the one to be tuned. This is known as"
                       " process variable (PV) in control theory. This comes from the process"
                       " we are trying to control." ),
        new DestFinfo( "gainDest", Ftype1< double >::global(),
                       RFCAST( &PIDController::setGain ),
                       "Destination message to control the PIDController gain dynamically." ),
    };
    static SchedInfo schedInfo[] = {{ process, 0, 0 }};
    static string doc[] = {
        "Name", "PIDController",
        "Author", "Subhasis Ray, 2008, NCBS",
        "Description", "PID feedback controller."
        "PID stands for Proportional-Integral-Derivative. It is used to"
        " feedback control dynamical systems. It tries to create a feedback"
        " output such that the sensed (measured) parameter is held at command"
        " value. Refer to wikipedia (http://wikipedia.org) for details on PID"
        " Controller." };

    static Cinfo pidCinfo(
            doc,
            sizeof( doc ) / sizeof( string ),
            initNeutralCinfo(),
            pidFinfos,
            sizeof( pidFinfos ) / sizeof( Finfo* ),
            ValueFtype1< PIDController >::global(),
            schedInfo, 1 );
    return &pidCinfo;
}

static const Cinfo* pidCinfo = initPIDControllerCinfo();
static const Slot outputSlot = initPIDControllerCinfo()->getSlot( "outputSrc" );

PIDController::PIDController():
        command_(0),
        saturation_(DBL_MAX),
        gain_(1),
        tau_i_(0),
        tau_d_(0),
        sensed_(0),
        output_(0),
        error_(0),
        e_integral_(0),
        e_derivative_(0),
        e_previous_(0)
{
    // do nothing else
}

void PIDController::setCommand( const Conn* conn, double command)
{
    PIDController* instance = static_cast< PIDController* >( conn->data() );
    instance->command_ = command;
}

double PIDController::getCommand( Eref e )
{
    PIDController* instance = static_cast< PIDController* >( e.e->data() );
    return instance->command_;
}

void PIDController::setSensed( const Conn* conn, double sensed )
{
    PIDController* instance = static_cast< PIDController* >( conn->data() );
    instance->sensed_ = sensed;
}

double PIDController::getSensed( Eref e )
{
    PIDController* instance = static_cast< PIDController* >( e.e->data() );
    return instance->sensed_;
}

double PIDController::getOutput( Eref e )
{
    PIDController* instance = static_cast< PIDController* >( e.e->data() );
    return instance->output_;
}

void PIDController::setGain( const Conn* conn, double gain )
{
    PIDController* instance = static_cast< PIDController* >( conn->data() );
    instance->gain_ = gain;
}

double PIDController::getGain( Eref e )
{
    PIDController* instance = static_cast< PIDController * >( e.e->data() );
    return instance->gain_;
}

void PIDController::setTauI( const Conn* conn, double tau_i )
{
    PIDController* instance = static_cast< PIDController* >( conn->data());
    instance->tau_i_ = tau_i;
}

double PIDController::getTauI( Eref e )
{
    PIDController* instance = static_cast< PIDController* >( e.e->data() );
    return instance->tau_i_;
}

void PIDController::setTauD( const Conn* conn, double tau_d )
{
    PIDController* instance = static_cast< PIDController* >( conn->data() );
    instance->tau_d_ = tau_d;
}

double PIDController::getTauD( Eref e )
{
    PIDController* instance = static_cast< PIDController* >( e.e->data() );
    return instance->tau_d_;
}

void PIDController::setSaturation( const Conn* conn, double saturation )
{
    PIDController* instance = static_cast< PIDController* >( conn->data() );
    if (saturation <= 0) {
        cout << "Error: PIDController::setSaturation - saturation must be positive." << endl;
    } else {
        instance->saturation_ = saturation;
    }
}

double PIDController::getSaturation( Eref e )
{
    PIDController* instance = static_cast< PIDController* >( e.e->data() );
    return instance->saturation_;
}

double PIDController::getError( Eref e )
{
    PIDController* instance = static_cast< PIDController* >( e.e->data() );
    return instance->error_;
}

double PIDController::getEIntegral( Eref e )
{
    PIDController* instance = static_cast< PIDController* >( e.e->data() );
    return instance->e_integral_;
}

double PIDController::getEDerivative( Eref e )
{
    PIDController* instance = static_cast< PIDController* >( e.e->data() );
    return instance->e_derivative_;
}

double PIDController::getEPrevious( Eref e )
{
    PIDController* instance = static_cast< PIDController* >( e.e->data() );
    return instance->e_previous_;
}

void PIDController::processFunc( const Conn* conn, ProcInfo proc )
{
    PIDController* instance = static_cast< PIDController* >( conn->data() );
    instance->e_previous_ = instance->error_;
    instance->error_ = instance->command_ - instance->sensed_;
    instance->e_integral_ += 0.5 * (instance->error_ + instance->e_previous_) * proc->dt_;
    instance->e_derivative_ = (instance->error_ - instance->e_previous_) / proc->dt_;
    instance->output_ = instance->gain_ * (instance->error_ +
                                           instance->e_integral_ / instance->tau_i_ +
                                           instance->e_derivative_ * instance->tau_d_);
    if (instance->output_ > instance->saturation_){
        instance->output_ = instance->saturation_;
        instance->e_integral_ -= 0.5 * (instance->error_ + instance->e_previous_) * proc->dt_;
    }
    else if (instance->output_ < -instance->saturation_){
        instance->output_ = -instance->saturation_;
        instance->e_integral_ -= 0.5 * (instance->error_ + instance->e_previous_) * proc->dt_;
    }
#ifndef NDEBUG
    cout << "PIDController::processFunc : " << conn->target().name() << ", command: " << instance->command_ << ", sensed: " << instance->sensed_ << ", e: " << instance->error_ << ", e_i: " << instance->e_integral_ << ", e_d: " << instance->e_derivative_ << ", e_prev: " << instance->e_previous_ << ", output: " << instance->output_ << ", gain: " << instance->gain_ << ", tauI: "<< instance->tau_i_ << ", tauD: " << instance->tau_d_ << endl;
#endif
    send1<double>( conn->target(), outputSlot, instance->output_);
}


void PIDController::reinitFunc( const Conn* conn, ProcInfo proc )
{
    PIDController* instance = static_cast< PIDController* >( conn->data());
    if ( instance->tau_i_ <= 0.0 )
        instance->tau_i_ = proc->dt_;
    if ( instance->tau_d_ < 0.0 )
        instance->tau_d_ = proc->dt_ / 4;
    instance->sensed_ = 0.0;
    instance->output_ = 0;
    instance->error_ = 0;
    instance->e_previous_ = instance->error_;
    instance->e_integral_ = 0;
    instance->e_derivative_ = 0;
}



// 
// PIDController.cpp ends here
