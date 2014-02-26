/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/
#include "header.h"
#ifdef USE_GSL
#include <gsl/gsl_errno.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_odeiv2.h>
#endif

#include "OdeSystem.h"
#include "VoxelPools.h"
#include "RateTerm.h"
#include "FuncTerm.h"
#include "SparseMatrix.h"
#include "KinSparseMatrix.h"
#include "Stoich.h"

//////////////////////////////////////////////////////////////
// Class definitions
//////////////////////////////////////////////////////////////

VoxelPools::VoxelPools()
	: 
		S_(1),
		Sinit_(1)
{
#ifdef USE_GSL
		driver_ = 0;
#endif
}

VoxelPools::~VoxelPools()
{
#ifdef USE_GSL
	if ( driver_ )
		gsl_odeiv2_driver_free( driver_ );
#endif
}

//////////////////////////////////////////////////////////////
// Array ops
//////////////////////////////////////////////////////////////
/// Using the computed array sizes, now allocate space for them.
void VoxelPools::resizeArrays( unsigned int totNumPools )
{
	S_.resize( totNumPools, 0.0 );
	Sinit_.resize( totNumPools, 0.0);
}

void VoxelPools::reinit()
{
	S_ = Sinit_;
}

//////////////////////////////////////////////////////////////
// Access functions
//////////////////////////////////////////////////////////////
const double* VoxelPools::S() const
{
	return &S_[0];
}

double* VoxelPools::varS()
{
	return &S_[0];
}

const double* VoxelPools::Sinit() const
{
	return &Sinit_[0];
}

double* VoxelPools::varSinit()
{
	return &Sinit_[0];
}

unsigned int VoxelPools::size() const
{
	return Sinit_.size();
}

//////////////////////////////////////////////////////////////
// Solver ops
//////////////////////////////////////////////////////////////
void VoxelPools::setStoich( const Stoich* s, const OdeSystem* ode )
{
	S_.resize( s->getNumAllPools(), 0.0 );
	Sinit_.resize( s->getNumAllPools(), 0.0);
#ifdef USE_GSL
	if ( driver_ )
		gsl_odeiv2_driver_free( driver_ );
	driver_ = gsl_odeiv2_driver_alloc_y_new( 
		&ode->gslSys, ode->gslStep, ode->initStepSize, 
		ode->epsAbs, ode->epsRel );
#endif
}

void VoxelPools::advance( const ProcInfo* p )
{
#ifdef USE_GSL
	double t = p->currTime;
	int status = gsl_odeiv2_driver_apply( driver_, &t, p->dt, &S_[0] );
	if ( status != GSL_SUCCESS ) {
		cout << "Error: VoxelPools::advance: GSL integration error at time "
			 << t << "\n";
		assert( 0 );
	}
#endif
}

//////////////////////////////////////////////////////////////
// Zombie Pool Access functions
//////////////////////////////////////////////////////////////

void VoxelPools::setN( unsigned int i, double v )
{
	S_[i] = v;
}

double VoxelPools::getN( unsigned int i ) const
{
	return S_[i];
}

void VoxelPools::setNinit( unsigned int i, double v )
{
	Sinit_[i] = v;
}

double VoxelPools::getNinit( unsigned int i ) const
{
	return Sinit_[i];
}

void VoxelPools::setDiffConst( unsigned int i, double v )
{
		; // Do nothing.
}

double VoxelPools::getDiffConst( unsigned int i ) const
{
		return 0;
}

