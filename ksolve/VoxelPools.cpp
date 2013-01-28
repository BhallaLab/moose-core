/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/
#include "header.h"
#include "../mesh/VoxelJunction.h"
#include "SolverJunction.h"
#include "VoxelPools.h"
#include "../shell/Shell.h"

//////////////////////////////////////////////////////////////
// Class definitions
//////////////////////////////////////////////////////////////

VoxelPools::VoxelPools()
	: 
		S_(1),
		Sinit_(1),
		solver_( 0 ) 

{;}

VoxelPools::~VoxelPools()
{;}

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

unsigned int VoxelPools::getSolver() const
{
	return solver_;
}

void VoxelPools::setSolver( unsigned int s )
{
	solver_ = s;
}


