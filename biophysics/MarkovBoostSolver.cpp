/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2011 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
#include "MarkovBoostSolver.h"
#include "../ksolve/BoostSys.h"

#include <functional>

using namespace std::placeholders;


#include <boost/numeric/odeint.hpp>

using namespace boost::numeric;

static SrcFinfo1< vector<double> >* stateOut()
{
    static SrcFinfo1< vector< double > > stateOut( "stateOut",
            "Sends updated state to the MarkovChannel class." );
    return &stateOut;
}

const Cinfo* MarkovBoostSolver::initCinfo()
{
    ///////////////////////////////////////////////////////
    // Field definitions
    ///////////////////////////////////////////////////////
    static ReadOnlyValueFinfo< MarkovBoostSolver, bool > isInitialized( 
            "isInitialized", 
            "True if the message has come in to set solver parameters.",
            &MarkovBoostSolver::getIsInitialized
            );
    static ValueFinfo< MarkovBoostSolver, string > method( "method", 
            "Numerical method to use.",
            &MarkovBoostSolver::setMethod,
            &MarkovBoostSolver::getMethod 
            );
    static ValueFinfo< MarkovBoostSolver, double > relativeAccuracy( 
            "relativeAccuracy", 
            "Accuracy criterion",
            &MarkovBoostSolver::setRelativeAccuracy,
            &MarkovBoostSolver::getRelativeAccuracy
            );
    static ValueFinfo< MarkovBoostSolver, double > absoluteAccuracy( 
            "absoluteAccuracy", 
            "Another accuracy criterion",
            &MarkovBoostSolver::setAbsoluteAccuracy,
            &MarkovBoostSolver::getAbsoluteAccuracy
            );
    static ValueFinfo< MarkovBoostSolver, double > internalDt( 
            "internalDt", 
            "internal timestep to use.",
            &MarkovBoostSolver::setInternalDt,
            &MarkovBoostSolver::getInternalDt
            );

    ///////////////////////////////////////////////////////
    // DestFinfo definitions
    ///////////////////////////////////////////////////////
    static DestFinfo init( "init",
            "Initialize solver parameters.",
            new OpFunc1< MarkovBoostSolver, vector< double > >
            ( &MarkovBoostSolver::init )
            );

    static DestFinfo handleQ( "handleQ",
            "Handles information regarding the instantaneous rate matrix from "
            "the MarkovRateTable class.",
            new OpFunc1< MarkovBoostSolver, vector< vector< double > > >( &MarkovBoostSolver::handleQ) );

    static DestFinfo process( "process",
            "Handles process call",
            new ProcOpFunc< MarkovBoostSolver >( &MarkovBoostSolver::process ) );
    static DestFinfo reinit( "reinit",
            "Handles reinit call",
            new ProcOpFunc< MarkovBoostSolver >( &MarkovBoostSolver::reinit ) );
    ///////////////////////////////////////////////////////
    // Shared definitions
    ///////////////////////////////////////////////////////
    static Finfo* procShared[] = {
        &process, &reinit
    };
    static SharedFinfo proc( "proc",
            "Shared message for process and reinit",
            procShared, sizeof( procShared ) / sizeof( const Finfo* )
            );

    static Finfo* MarkovBoostFinfos[] =
    {
        &isInitialized,			// ValueFinfo
        &method,						// ValueFinfo
        &relativeAccuracy,	// ValueFinfo
        &absoluteAccuracy,	// ValueFinfo
        &internalDt,				// ValueFinfo
        &init,							// DestFinfo
        &handleQ,						// DestFinfo	
        &proc,							// SharedFinfo
        stateOut(),  				// SrcFinfo
    };

    static string doc[] = 
    {
        "Name", "MarkovBoostSolver",
        "Author", "Vishaka Datta S, 2011, NCBS",
        "Description", "Solver for Markov Channel." 
    };

    static Dinfo< MarkovBoostSolver > dinfo;
    static Cinfo MarkovBoostSolverCinfo(
            "MarkovBoostSolver",
            Neutral::initCinfo(),
            MarkovBoostFinfos,
            sizeof(MarkovBoostFinfos)/sizeof(Finfo *),
            &dinfo,
            doc,
            sizeof(doc) / sizeof(string)
            );

    return &MarkovBoostSolverCinfo;
}

static const Cinfo* MarkovBoostSolverCinfo = MarkovBoostSolver::initCinfo();

///////////////////////////////////////////////////
// Basic class function definitions
///////////////////////////////////////////////////

MarkovBoostSolver::MarkovBoostSolver()
{
    isInitialized_ = 0;
    method_ = "rk5";
    boostStepType_ = 1;                         /* FIXME: Use enum */
    boostStep_ = 0;
    nVars_ = 0;
    absAccuracy_ = 1.0e-8;
    relAccuracy_ = 1.0e-8;
    internalStepSize_ = 1.0e-6;
    stateBoost_ = 0;
}

MarkovBoostSolver::~MarkovBoostSolver()
{
}

int MarkovBoostSolver::evalSystem( const vector<double>& state, vector<double>& f, double t 
        , vector< vector< double > >& Q)
{
    unsigned int nVars = state.size();

    //Matrix being accessed along columns, which is a very bad thing in terms of
    //cache optimality. Transposing the matrix during reinit() would be a good idea.
    for ( unsigned int i = 0; i < nVars; ++i)
    {
        f[i] = 0;
        for ( unsigned int j = 0; j < nVars; ++j)
            f[i] += state[j] * (Q[j][i]);
    }

    return 0;
}

///////////////////////////////////////////////////
// Field function definitions
///////////////////////////////////////////////////

bool MarkovBoostSolver::getIsInitialized() const
{
    return isInitialized_;
}

string MarkovBoostSolver::getMethod() const
{
    return method_;
}

void MarkovBoostSolver::setMethod( string method )
{
    method_ = method;
    boostStepType_ = 0;

#if 0
    if ( method == "rk2" ) {
        boostStepType_ = "rk
    } else if ( method == "rk4" ) {
        boostStepType_ = gsl_odeiv_step_rk4;
    } else if ( method == "rk5" ) {
        boostStepType_ = gsl_odeiv_step_rkf45;
    } else if ( method == "rkck" ) {
        boostStepType_ = gsl_odeiv_step_rkck;
    } else if ( method == "rk8pd" ) {
        boostStepType_ = gsl_odeiv_step_rk8pd;
    } else if ( method == "rk2imp" ) {
        boostStepType_ = gsl_odeiv_step_rk2imp;
    } else if ( method == "rk4imp" ) {
        boostStepType_ = gsl_odeiv_step_rk4imp;
    } else if ( method == "bsimp" ) {
        boostStepType_ = gsl_odeiv_step_rk4imp;
        cout << "Warning: implicit Bulirsch-Stoer method not yet implemented: needs Jacobian\n";
    } else if ( method == "gear1" ) {
        boostStepType_ = gsl_odeiv_step_gear1;
    } else if ( method == "gear2" ) {
        boostStepType_ = gsl_odeiv_step_gear2;
    } else {
        cout << "Warning: MarkovBoostSolver::innerSetMethod: method '" <<
            method << "' not known, using rk5\n";
        boostStepType_ = gsl_odeiv_step_rkf45;
    }
#endif

}

double MarkovBoostSolver::getRelativeAccuracy() const
{
    return relAccuracy_;
}

void MarkovBoostSolver::setRelativeAccuracy( double value )
{
    relAccuracy_ = value;
}

double MarkovBoostSolver::getAbsoluteAccuracy() const
{
    return absAccuracy_;
}
void MarkovBoostSolver::setAbsoluteAccuracy( double value )
{
    absAccuracy_ = value;
}

double MarkovBoostSolver::getInternalDt() const
{
    return internalStepSize_;
}

void MarkovBoostSolver::setInternalDt( double value )
{
    internalStepSize_ = value;
}

///////////////////////////////////////////////////
// Dest function definitions
///////////////////////////////////////////////////

//Handles data from MarkovChannel class.
void MarkovBoostSolver::init( vector< double > initialState )
{
    nVars_ = initialState.size();

    if ( stateBoost_ == 0 )
        stateBoost_ = new double[ nVars_ ];

    state_ = initialState;
    initialState_ = initialState;

    Q_.resize( nVars_ );

    for ( unsigned int i = 0; i < nVars_; ++i )
        Q_[i].resize( nVars_, 0.0 );	

    isInitialized_ = 1;

    assert( boostStepType_ != 0 );
}

//////////////////////////
//MsgDest functions.
/////////////////////////
void MarkovBoostSolver::process( const Eref& e, ProcPtr info )
{
    double nextt = info->currTime + info->dt;
    double t = info->currTime;
    double sum = 0;

    for ( unsigned int i = 0; i < nVars_; ++i )
        stateBoost_[i] = state_[i];

    auto stepper = odeint::make_controlled< rk_dopri_stepper_type_ >( 
            absAccuracy_ , relAccuracy_ 
            );

    {
        system = std::bind( &MarkovBoostSolver::evalSystem , _1, _2, _3, Q_ );

        int status = odeint::integrate_adaptive ( stepper, system, stateBoost_
                , t , nextt, info->dt
                );

        //Simple idea borrowed from Dieter Jaeger's implementation of a Markov
        //channel to deal with potential round-off error.
        sum = 0;
        for ( unsigned int i = 0; i < nVars_; i++ )
            sum += stateBoost_[i];

        for ( unsigned int i = 0; i < nVars_; i++ )
            stateBoost_[i] /= sum;

    }

    for ( unsigned int i = 0; i < nVars_; ++i )
        state_[i] = stateBoost_[i];

    stateOut()->send( e, state_ );
}

void MarkovBoostSolver::reinit( const Eref& e, ProcPtr info )
{
    state_ = initialState_;
    if ( initialState_.empty() )
    {
        cerr << "MarkovBoostSolver::reinit : "
            "Initial state has not been set. Solver has not been initialized."
            "Call init() before running.\n";
    }

    stateOut()->send( e, state_ );
}

void MarkovBoostSolver::handleQ( vector< vector< double > > Q )
{
    Q_ = Q;
}
