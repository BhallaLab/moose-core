/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2014 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/
#include "header.h"
#include "SparseMatrix.h"
#include "ZombiePoolInterface.h"
#include "KinSparseMatrix.h"
#include "DiffPoolVec.h"

#define EPSILON 1e-15

/*
const Cinfo* DiffPoolVec::initCinfo()
{
		//////////////////////////////////////////////////////////////
		// Field Definitions:
		//////////////////////////////////////////////////////////////
		Static ReadOnlyValueFinfo< DiffPoolVec, Id > poolId(
			"poolId",
			"Identifies which pool is handled by this DiffPoolVec"
			&DiffPoolVec::getPoolId
		);

		//////////////////////////////////////////////////////////////
		// MsgDest Definitions:
		//////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////
		// SrcFinfo Definitions: All inherited.
		//////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////
		// SharedMsg Definitions: All inherited.
		//////////////////////////////////////////////////////////////
	static Finfo* poolFinfos[] = {
		&increment,			// DestFinfo
		&decrement,			// DestFinfo
	};

	static Dinfo< Pool > dinfo;
	static Cinfo poolCinfo (
		"Pool",
		PoolBase::initCinfo(),
		poolFinfos,
		sizeof( poolFinfos ) / sizeof ( Finfo* ),
		&dinfo
	);

	return &poolCinfo;
}
*/

/**
 * Default is to create it with a single compartment, independent of any
 * solver, so that we can set it up as a dummy DiffPool for the Pool to
 * work on in single-compartment models.
 */
DiffPoolVec::DiffPoolVec()
	: n_( 1, 0.0 ), nInit_( 1, 0.0 ), diffConst_( 1.0e-12 )
{;}

double DiffPoolVec::getNinit( const Eref& e ) const
{
	unsigned int voxel = e.dataIndex();
	assert( voxel < nInit_.size() );
	return nInit_[ voxel ];
}

void DiffPoolVec::setNinit( const Eref& e, double v )
{
	unsigned int voxel = e.dataIndex();
	assert( voxel < nInit_.size() );
	nInit_[ voxel ] = v;
}

double DiffPoolVec::getN( const Eref& e ) const
{
	unsigned int voxel = e.dataIndex();
	assert( voxel < n_.size() );
	return n_[ voxel ];
}

void DiffPoolVec::setN( const Eref& e, double v )
{
	unsigned int voxel = e.dataIndex();
	assert( voxel < n_.size() );
	n_[ voxel ] = v;
}

double DiffPoolVec::getDiffConst() const
{
	return diffConst_;
}

void DiffPoolVec::setDiffConst( double v )
{
	diffConst_ = v;
}

void DiffPoolVec::process() // Not called by the clock, but by parent.
{
	if ( ops_.size() > 0 )
		advance();
}

void DiffPoolVec::reinit() // Not called by the clock, but by parent.
{
	assert( n_.size() == nInit_.size() );
	n_ = nInit_;
}

void DiffPoolVec::setPool( Id pool )
{
	pool_ = pool;
}

Id DiffPoolVec::getPool() const
{
	return pool_;
}

void DiffPoolVec::setOps(const vector< Triplet< double > >& ops,
	const vector< double >& diagVal )
{
	assert( diagVal_.size() == n_.size() );
	ops_ = ops;
	diagVal_ = diagVal;
}

void DiffPoolVec::advance()
{
	for ( vector< Triplet< double > >::const_iterator
				i = ops_.begin(); i != ops_.end(); ++i )
		n_[i->c_] -= n_[i->b_] * i->a_;

	assert( n_.size() == diagVal_.size() );
	vector< double >::iterator iy = n_.begin();
	for ( vector< double >::const_iterator
				i = diagVal_.begin(); i != diagVal_.end(); ++i )
		*iy++ *= *i;
}
