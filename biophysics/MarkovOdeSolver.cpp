/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2011 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "../basecode/header.h"
#include "../utility/boost_ode.h"
#include "MarkovOdeSolver.h"
#include "boost/numeric/odeint.hpp"
using namespace boost::numeric;

static SrcFinfo1< vector<double> >* stateOut()
{
    static SrcFinfo1< vector< double > > stateOut( "stateOut",
            "Sends updated state to the MarkovChannel class." );
    return &stateOut;
}

const Cinfo* MarkovOdeSolver::initCinfo()
{
    ///////////////////////////////////////////////////////
    // Field definitions
    ///////////////////////////////////////////////////////
    static ReadOnlyValueFinfo< MarkovOdeSolver, bool > isInitialized(
        "isInitialized",
        "True if the message has come in to set solver parameters.",
        &MarkovOdeSolver::getIsInitialized
    );
    static ValueFinfo< MarkovOdeSolver, string > method( "method",
            "Numerical method to use.",
            &MarkovOdeSolver::setMethod,
            &MarkovOdeSolver::getMethod
                                                       );
    static ValueFinfo< MarkovOdeSolver, double > relativeAccuracy(
        "relativeAccuracy",
        "Accuracy criterion",
        &MarkovOdeSolver::setRelativeAccuracy,
        &MarkovOdeSolver::getRelativeAccuracy
    );
    static ValueFinfo< MarkovOdeSolver, double > absoluteAccuracy(
        "absoluteAccuracy",
        "Another accuracy criterion",
        &MarkovOdeSolver::setAbsoluteAccuracy,
        &MarkovOdeSolver::getAbsoluteAccuracy
    );
    static ValueFinfo< MarkovOdeSolver, double > internalDt(
        "internalDt",
        "internal timestep to use.",
        &MarkovOdeSolver::setInternalDt,
        &MarkovOdeSolver::getInternalDt
    );

    ///////////////////////////////////////////////////////
    // DestFinfo definitions
    ///////////////////////////////////////////////////////
    static DestFinfo init( "init",
                           "Initialize solver parameters.",
                           new OpFunc1< MarkovOdeSolver, vector< double > >
                           ( &MarkovOdeSolver::init )
                         );

    static DestFinfo handleQ( "handleQ",
                              "Handles information regarding the instantaneous rate matrix from "
                              "the MarkovRateTable class.",
                              new OpFunc1< MarkovOdeSolver, vector< vector< double > > >( &MarkovOdeSolver::handleQ) );

    static DestFinfo process( "process",
                              "Handles process call",
                              new ProcOpFunc< MarkovOdeSolver >( &MarkovOdeSolver::process ) );
    static DestFinfo reinit( "reinit",
                             "Handles reinit call",
                             new ProcOpFunc< MarkovOdeSolver >( &MarkovOdeSolver::reinit ) );
    ///////////////////////////////////////////////////////
    // Shared definitions
    ///////////////////////////////////////////////////////
    static Finfo* procShared[] =
    {
        &process, &reinit
    };
    static SharedFinfo proc( "proc",
                             "Shared message for process and reinit",
                             procShared, sizeof( procShared ) / sizeof( const Finfo* )
                           );

    static Finfo* MarkovOdeFinfos[] =
    {
        &isInitialized,         // ValueFinfo
        &method,                // ValueFinfo
        &relativeAccuracy,      // ValueFinfo
        &absoluteAccuracy,      // ValueFinfo
        &internalDt,            // ValueFinfo
        &init,                  // DestFinfo
        &handleQ,               // DestFinfo
        &proc,                  // SharedFinfo
        stateOut(),             // SrcFinfo
    };

    static string doc[] =
    {
        "Name", "MarkovOdeSolver",
        "Author", "Vishaka Datta (c) 2011, Dilawar Singh (c) 2018",
        "Description", "Solver for Markov Channel."
    };

    static Dinfo< MarkovOdeSolver > dinfo;
    static Cinfo MarkovOdeSolverCinfo(
        "MarkovOdeSolver",
        Neutral::initCinfo(),
        MarkovOdeFinfos,
        sizeof(MarkovOdeFinfos)/sizeof(Finfo *),
        &dinfo,
        doc,
        sizeof(doc) / sizeof(string)
    );

    return &MarkovOdeSolverCinfo;
}

static const Cinfo* MarkovOdeSolverCinfo = MarkovOdeSolver::initCinfo();

///////////////////////////////////////////////////
// Basic class function definitions
///////////////////////////////////////////////////

MarkovOdeSolver::MarkovOdeSolver()
{
    isInitialized_ = 0;
    method_ = "rk5";
    nVars_ = 0;
    absAccuracy_ = 1e-8;
    relAccuracy_ = 1e-8;
    internalStepSize_ = 1.0e-6;
}

MarkovOdeSolver::~MarkovOdeSolver()
{
}

/* --------------------------------------------------------------------------*/
/**
 * @Synopsis  OdeSystem. Similar to MarkovOdeSolver::evalSystem
 *
 * @Param y
 * @Param dydt
 */
/* ----------------------------------------------------------------------------*/
void MarkovOdeSolver::OdeSystem( const vector<double>& y, vector<double>& dydt )
{
    unsigned int nVars = Q_.size();
    //Matrix being accessed along columns, which is a very bad thing in terms of
    //cache optimality. Transposing the matrix during reinit() would be a good idea.
    for ( unsigned int i = 0; i < nVars; ++i)
    {
        dydt[i] = 0;
        for ( unsigned int j = 0; j < nVars; ++j)
            dydt[i] += y[j] * Q_[j][i];
    }
}

///////////////////////////////////////////////////
// Field function definitions
///////////////////////////////////////////////////

bool MarkovOdeSolver::getIsInitialized() const
{
    return isInitialized_;
}

string MarkovOdeSolver::getMethod() const
{
    return method_;
}

void MarkovOdeSolver::setMethod( string method )
{
    method_ = method;
}

double MarkovOdeSolver::getRelativeAccuracy() const
{
    return relAccuracy_;
}

void MarkovOdeSolver::setRelativeAccuracy( double value )
{
    relAccuracy_ = value;
}

double MarkovOdeSolver::getAbsoluteAccuracy() const
{
    return absAccuracy_;
}
void MarkovOdeSolver::setAbsoluteAccuracy( double value )
{
    absAccuracy_ = value;
}

double MarkovOdeSolver::getInternalDt() const
{
    return internalStepSize_;
}

void MarkovOdeSolver::setInternalDt( double value )
{
    internalStepSize_ = value;
}

///////////////////////////////////////////////////
// Dest function definitions
///////////////////////////////////////////////////

//Handles data from MarkovChannel class.
void MarkovOdeSolver::init( vector< double > initialState )
{

    nVars_ = initialState.size();
    state_ = initialState;
    initialState_ = initialState;
    Q_.resize( nVars_ );
    for ( unsigned int i = 0; i < nVars_; ++i )
        Q_[i].resize( nVars_, 0.0 );

    stateOde_.resize( nVars_ );
    isInitialized_ = 1;


}

//////////////////////////
//MsgDest functions.
/////////////////////////
void MarkovOdeSolver::process( const Eref& e, ProcPtr p )
{
    double nextt = p->currTime + p->dt;
    double t = p->currTime;
    double sum = 0;

    for ( unsigned int i = 0; i < nVars_; ++i )
        stateOde_[i] = state_[i];

    auto sys = [this](const vector<double>& dy, vector<double>& dydt, const double t)
    { 
        this->OdeSystem(dy, dydt); 
    };

    // It works well in practice for this setup. 
    if( method_ == "rk5" || method_ == "gsl" || method_ == "boost" )
        odeint::integrate( sys, stateOde_, t, nextt, p->dt );
    else if( method_ == "rk5a" || method_ == "adaptive" )
        odeint::integrate_adaptive( 
                odeint::make_controlled<rk_dopri_stepper_type_>( absAccuracy_, relAccuracy_ )
                , sys, stateOde_, t, nextt, p->dt
                );
    else if( method_ == "rk4" )
        odeint::integrate_const( rk4_stepper_type_()
                , sys, stateOde_, t, nextt, p->dt
                );
    else if ("rk54" == method_ )
        odeint::integrate_const( rk_karp_stepper_type_()
                , sys, stateOde_, t, nextt, p->dt
                );
    else if ("rkck" == method_ || "rkcka" == method_)
        odeint::integrate_adaptive(
                odeint::make_controlled<rk_karp_stepper_type_>( absAccuracy_, relAccuracy_ )
                , sys, stateOde_, t, nextt, p->dt
                );
    else if( method_ == "rk8a" || "rk8" == method_ )
        odeint::integrate_adaptive( rk_felhberg_stepper_type_()
                , sys, stateOde_, t, nextt, p->dt
                );
    else
        odeint::integrate( sys, stateOde_, t, nextt, p->dt );


    for ( unsigned int i = 0; i < nVars_; ++i )
        state_[i] = stateOde_[i];

    stateOut()->send( e, state_ );
}

void MarkovOdeSolver::reinit( const Eref& e, ProcPtr info )
{
    state_ = initialState_;
    if ( initialState_.empty() )
    {
        cerr << "MarkovOdeSolver::reinit : "
             "Initial state has not been set. Solver has not been initialized."
             "Call init() before running.\n";
    }
    stateOut()->send( e, state_ );
}

void MarkovOdeSolver::handleQ( vector< vector< double > > Q )
{
    Q_ = Q;
}
