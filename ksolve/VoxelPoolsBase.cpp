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
void VoxelPoolsBase::xferIn(
		const vector< unsigned int >& poolIndex,
		const vector< double >& values, 
		const vector< double >& lastValues,
	    unsigned int voxelIndex	)
{
	unsigned int offset = voxelIndex * poolIndex.size();
	vector< double >::const_iterator i = values.begin() + offset;
	vector< double >::const_iterator j = lastValues.begin() + offset;
	for ( vector< unsigned int >::const_iterator 
			k = poolIndex.begin(); k != poolIndex.end(); ++k ) {
		S_[*k] += *i++ - *j++;
	}
}

void VoxelPoolsBase::xferInOnlyProxies(
		const vector< unsigned int >& poolIndex,
		const vector< double >& values, 
		unsigned int numProxyPools,
	    unsigned int voxelIndex	)
{
	unsigned int offset = voxelIndex * poolIndex.size();
	vector< double >::const_iterator i = values.begin() + offset;
	for ( vector< unsigned int >::const_iterator 
			k = poolIndex.begin(); k != poolIndex.end(); ++k ) {
		if ( *k >= S_.size() - numProxyPools ) {
			Sinit_[*k] = S_[*k] += *i;
		}
		i++;
	}
}

void VoxelPoolsBase::xferOut( 
	unsigned int voxelIndex, 
	vector< double >& values,
	const vector< unsigned int >& poolIndex)
{
	unsigned int offset = voxelIndex * poolIndex.size();
	vector< double >::iterator i = values.begin() + offset;
	for ( vector< unsigned int >::const_iterator 
			k = poolIndex.begin(); k != poolIndex.end(); ++k ) {
		*i++ = S_[*k];
	}
}

void VoxelPoolsBase::addProxyVoxy( 
				unsigned int comptIndex, unsigned int voxel )
{
	if ( comptIndex >= proxyPoolVoxels_.size() )
		proxyPoolVoxels_.resize( comptIndex + 1 );
	proxyPoolVoxels_[comptIndex].push_back( voxel );
}

void VoxelPoolsBase::addProxyTransferIndex( 
				unsigned int comptIndex, unsigned int transferIndex )
{
	if ( comptIndex >= proxyTransferIndex_.size() )
		proxyTransferIndex_.resize( comptIndex + 1 );
	proxyTransferIndex_[comptIndex].push_back( transferIndex );
}

bool VoxelPoolsBase::hasXfer( unsigned int comptIndex ) const
{
	if ( comptIndex >= proxyPoolVoxels_.size() )
		return false;
	return (proxyPoolVoxels_[ comptIndex ].size() > 0);
}
