/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/
#include "header.h"

#include "VoxelPools.h"

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

void VoxelPools::setSolver( const Ksolve* s )
{
	ksolve_ = s;
}

unsigned int VoxelPools::getPoolIndex( const Eref& e )
{
	unsigned int pool = ksolve_->convertIdToReacIndex( e.id() );
	assert( pool < S_.size() );
	return pool;
}

//////////////////////////////////////////////////////////////
// Zombie Pool Access functions
//////////////////////////////////////////////////////////////

void VoxelPools::setN( const Eref& e, double v )
{
	S_[ getPoolIndex( e ) ] = v;
}

double VoxelPools::getN( const Eref& e ) const
{
	return S_[ getPoolIndex( e ) ];
}

void VoxelPools::setNinit( const Eref& e, double v )
{
	Sinit_[ getPoolIndex( e ) ] = v;
}

double VoxelPools::getNinit( const Eref& e ) const
{
	return Sinit_[  getPoolIndex( e )  ];
}

void VoxelPools::setDiffConst( const Eref& e, double v )
{
		; // Do nothing.
}

double VoxelPools::getDiffConst( const Eref& e ) const
{
		return 0;
}

