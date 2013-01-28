
/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/
#include "StoichHeaders.h"
#include <gsl/gsl_errno.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_odeiv.h>
#include "OdeSystem.h"

OdeSystem::OdeSystem()
	: 
	stoich_( 0 ),
	gslStepType_( gsl_odeiv_step_rkf45 ), 
	gslStep_( 0 ), 
	gslControl_( 0 ), 
	gslEvolve_( 0 )
{;}

OdeSystem::OdeSystem( const StoichCore* master,
					const vector< Id >& compartmentSignature )
	: 
	stoich_( master->spawn( compartmentSignature ) ),
	compartmentSignature_( compartmentSignature ),
	gslStepType_( gsl_odeiv_step_rkf45 ), 
	gslStep_( 0 ), 
	gslControl_( 0 ), 
	gslEvolve_( 0 )
{;}

OdeSystem::~OdeSystem()
{
		/*
	if ( gslEvolve_ )
		gsl_odeiv_evolve_free( gslEvolve_ );
	if ( gslControl_ )
		gsl_odeiv_control_free( gslControl_ );
	if ( gslStep_ )
		gsl_odeiv_step_free( gslStep_ );
		*/
}

void OdeSystem::reallyFreeOdeSystem()
{
	if ( gslEvolve_ )
		gsl_odeiv_evolve_free( gslEvolve_ );
	if ( gslControl_ )
		gsl_odeiv_control_free( gslControl_ );
	if ( gslStep_ )
		gsl_odeiv_step_free( gslStep_ );
	delete stoich_;
}

string OdeSystem::setMethod( const string& method )
{
	gslStepType_ = 0;
	// cout << "in void GslStoich::innerSetMethod( string method ) \n";
	if ( method == "rk2" ) {
		gslStepType_ = gsl_odeiv_step_rk2;
	} else if ( method == "rk4" ) {
		gslStepType_ = gsl_odeiv_step_rk4;
	} else if ( method == "rk5" || method == "gsl" ) {
		gslStepType_ = gsl_odeiv_step_rkf45;
		return "rk5";
	} else if ( method == "rkck" ) {
		gslStepType_ = gsl_odeiv_step_rkck;
	} else if ( method == "rk8pd" ) {
		gslStepType_ = gsl_odeiv_step_rk8pd;
	} else if ( method == "rk2imp" ) {
		gslStepType_ = gsl_odeiv_step_rk2imp;
	} else if ( method == "rk4imp" ) {
		gslStepType_ = gsl_odeiv_step_rk4imp;
	} else if ( method == "bsimp" ) {
		gslStepType_ = gsl_odeiv_step_rk4imp;
		cout << "Warning: implicit Bulirsch-Stoer method not yet implemented: needs Jacobian. Using rk5\n";
		gslStepType_ = gsl_odeiv_step_rkf45;
		return "rk5";
	} else if ( method == "gear1" ) {
		gslStepType_ = gsl_odeiv_step_gear1;
	} else if ( method == "gear2" ) {
		gslStepType_ = gsl_odeiv_step_gear2;
	} else {
		cout << "Warning: OdeSystem::innerSetMethod: method '" <<
			method << "' not known, using rk5\n";
		gslStepType_ = gsl_odeiv_step_rkf45;
		return "rk5";
	}
	return method;
}

void OdeSystem::reinit( 
		void* gslStoich,
		int func (double t, const double *y, double *f, void *params),
		unsigned int nVarPools, double absAccuracy, double relAccuracy )
{
	assert( stoich_ != 0 );
	assert( gslStepType_ != 0 );
	if ( gslStep_ )
		gsl_odeiv_step_free(gslStep_);
	gslStep_ = gsl_odeiv_step_alloc( gslStepType_, nVarPools );
   	assert( gslStep_ != 0 );
        
	if ( gslEvolve_ )
		gsl_odeiv_evolve_free( gslEvolve_ );
	gslEvolve_ = gsl_odeiv_evolve_alloc(nVarPools);
	assert(gslEvolve_ != 0);
        
	if ( !gslControl_ )
		gslControl_ = gsl_odeiv_control_y_new( absAccuracy, relAccuracy );
	else 
		gsl_odeiv_control_init( 
							gslControl_, absAccuracy, relAccuracy, 1, 0 );
	assert( gslControl_!= 0 );
	gslSys_.params = gslStoich;
	gslSys_.function = func;
	gslSys_.jacobian = 0;
	gslSys_.dimension = nVarPools;
}
