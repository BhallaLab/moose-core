/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
#include "PoolBase.h"
#include "ZombiePoolInterface.h"
#include "ZombiePool.h"
#include "lookupVolumeFromMesh.h"
#include "ElementValueFinfo.h"

#define EPSILON 1e-15

const Cinfo* ZombiePool::initCinfo()
{
		//////////////////////////////////////////////////////////////
		// Field Definitions: All inherited from PoolBase
		//////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////
		// MsgDest Definitions: All inherited from PoolBase
		//////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////
		// SrcFinfo Definitions: All inherited from PoolBase
		//////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////
		// SharedMsg Definitions: All inherited from PoolBase
		//////////////////////////////////////////////////////////////

	// Note that here the isOneZombie_ flag on the Dinfo constructor is
	// true. This means that the duplicate and copy operations only make
	// one copy, regardless of how big the array of zombie pools.
	// The assumption is that each Id has a single pool, which can be
	// present in many voxels.
	static Dinfo< ZombiePool > dinfo( true );
	static Cinfo zombiePoolCinfo (
		"ZombiePool",
		PoolBase::initCinfo(),
		0,
		0,
		&dinfo
	);

	return &zombiePoolCinfo;
}




//////////////////////////////////////////////////////////////
// Class definitions
//////////////////////////////////////////////////////////////
static const Cinfo* zombiePoolCinfo = ZombiePool::initCinfo();

ZombiePool::ZombiePool()
	: dsolve_( 0 ), ksolve_( 0 ), diffConst_( 1e-12 )
{;}

ZombiePool::~ZombiePool()
{;}

//////////////////////////////////////////////////////////////
// MsgDest Definitions
//////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////
// Field Definitions
//////////////////////////////////////////////////////////////

void ZombiePool::vSetN( const Eref& e, double v )
{
	if ( ksolve_ )
		ksolve_->setN( e, v );
	if ( dsolve_ )
		dsolve_->setN( e, v );
	// S_[ e.index().value() ][ convertIdToPoolIndex( e.id() ) ] = v;
}

double ZombiePool::vGetN( const Eref& e ) const
{
	if ( dsolve_ != 0 )
		return dsolve_->getN( e );
	else if ( ksolve_ != 0 )
		return ksolve_->getN( e );
	return 0.0;
}

void ZombiePool::vSetNinit( const Eref& e, double v )
{
	if ( ksolve_ )
		ksolve_->setNinit( e, v );
	if ( dsolve_ )
		dsolve_->setNinit( e, v );
}

double ZombiePool::vGetNinit( const Eref& e ) const
{
	if ( dsolve_ != 0 )
		return dsolve_->getNinit( e );
	else if ( ksolve_ != 0 )
		return ksolve_->getNinit( e );
	return 0.0;
}

void ZombiePool::vSetConc( const Eref& e, double conc )
{
	// unsigned int pool = convertIdToPoolIndex( e.id() );
	double n = NA * conc * lookupVolumeFromMesh( e );
	vSetN( e, n );
}

double ZombiePool::vGetConc( const Eref& e ) const
{
	return vGetN( e ) / ( NA * lookupVolumeFromMesh( e ) );
}

void ZombiePool::vSetConcInit( const Eref& e, double conc )
{
	double n = NA * conc * lookupVolumeFromMesh( e );
	vSetNinit( e, n );
}

double ZombiePool::vGetConcInit( const Eref& e ) const
{
	return vGetNinit( e ) / ( NA * lookupVolumeFromMesh( e ) );
}

void ZombiePool::vSetDiffConst( const Eref& e, double v )
{
	if ( dsolve_ )
		dsolve_->setDiffConst( v );
	diffConst_ = v;
}

double ZombiePool::vGetDiffConst( const Eref& e ) const
{
	return diffConst_;
}

void ZombiePool::vSetSpecies( const Eref& e, unsigned int v )
{
	;
}

unsigned int ZombiePool::vGetSpecies( const Eref& e ) const
{
	return 0;
}

void ZombiePool::vSetVolume( const Eref& e, double v )
{
	assert( 0 ); // Illegal op, but should handle more gracefully.
}

double ZombiePool::vGetVolume( const Eref& e ) const
{
	return lookupVolumeFromMesh( e );
}

//////////////////////////////////////////////////////////////
// Zombie conversion functions.
//////////////////////////////////////////////////////////////

void ZombiePool::setSolver( Id s )
{
	// Nasty unsafe typecast. I would have preferred to pass in a 
	// safely typed pointer but that would have exposed a low-level
	// class for the ZombiePoolInterface.
	ZombiePoolInterface* zpi = reinterpret_cast< ZombiePoolInterface *>(
					ObjId( s, 0 ).data() );

	if ( dsolve_ == 0 )
		dsolve_ = zpi;
	else
		ksolve_ = zpi;
}

/*
void ZombiePool::setKineticSolver( ZombiePoolInterface* k )
{
	assert( k != 0 );
	ksolve_ = k ;
}

void ZombiePool::setDiffSolver( ZombiePoolInterface* d )
{
	assert( d != 0 );
	dsolve_ = d;
}
*/
