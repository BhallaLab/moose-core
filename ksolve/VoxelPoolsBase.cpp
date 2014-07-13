/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/
#include "header.h"

#include "VoxelPoolsBase.h"

//////////////////////////////////////////////////////////////
// Class definitions
//////////////////////////////////////////////////////////////

VoxelPoolsBase::VoxelPoolsBase()
	: 
		S_(1),
		Sinit_(1),
		volume_(1.0)
{;}

VoxelPoolsBase::~VoxelPoolsBase()
{}

//////////////////////////////////////////////////////////////
// Array ops
//////////////////////////////////////////////////////////////
/// Using the computed array sizes, now allocate space for them.
void VoxelPoolsBase::resizeArrays( unsigned int totNumPools )
{
	S_.resize( totNumPools, 0.0 );
	Sinit_.resize( totNumPools, 0.0);
}

void VoxelPoolsBase::reinit()
{
	S_ = Sinit_;
}

//////////////////////////////////////////////////////////////
// Access functions
//////////////////////////////////////////////////////////////
const double* VoxelPoolsBase::S() const
{
	return &S_[0];
}

vector< double >& VoxelPoolsBase::Svec()
{
	return S_;
}

double* VoxelPoolsBase::varS()
{
	return &S_[0];
}

const double* VoxelPoolsBase::Sinit() const
{
	return &Sinit_[0];
}

double* VoxelPoolsBase::varSinit()
{
	return &Sinit_[0];
}

unsigned int VoxelPoolsBase::size() const
{
	return Sinit_.size();
}

void VoxelPoolsBase::setVolume( double vol )
{
	volume_ = vol;
}

double VoxelPoolsBase::getVolume() const
{
	return volume_;
}

//////////////////////////////////////////////////////////////
// Zombie Pool Access functions
//////////////////////////////////////////////////////////////

void VoxelPoolsBase::setN( unsigned int i, double v )
{
	S_[i] = v;
}

double VoxelPoolsBase::getN( unsigned int i ) const
{
	return S_[i];
}

void VoxelPoolsBase::setNinit( unsigned int i, double v )
{
	Sinit_[i] = v;
}

double VoxelPoolsBase::getNinit( unsigned int i ) const
{
	return Sinit_[i];
}

void VoxelPoolsBase::setDiffConst( unsigned int i, double v )
{
		; // Do nothing.
}

double VoxelPoolsBase::getDiffConst( unsigned int i ) const
{
		return 0;
}

//////////////////////////////////////////////////////////////
// Handle cross compartment reactions
//////////////////////////////////////////////////////////////
void VoxelPoolsBase::mergeProxy(
		const vector< double >& remote, vector< double >& local )
{
	for ( vector< pair< int, int > >::const_iterator
		i = xfer_.begin(); i != xfer_.end(); ++i ) {
		S_[i->first] = local[i->second] = 
			S_[i->first ] + remote[i->second] - local[i->second];
	}
}

void VoxelPoolsBase::setXfer( const vector< pair< int, int > >& xfer )
{
	xfer_ = xfer;
}
