/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

// #include "StoichHeaders.h"
#include "header.h"
#include "../mesh/VoxelJunction.h"
#include "SolverJunction.h"
#include "SolverBase.h"
#include "../kinetics/PoolBase.h"
#include "ZPool.h"
#include "lookupSizeFromMesh.h"
// #include "ElementValueFinfo.h"

#define EPSILON 1e-15

const Cinfo* ZPool::initCinfo()
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
	// one copy, regardless of how many dims they are requested to do.
	static Cinfo zombiePoolCinfo (
		"ZPool",
		PoolBase::initCinfo(),
		0,
		0,
		new Dinfo< ZPool >( true )
	);

	return &zombiePoolCinfo;
}




//////////////////////////////////////////////////////////////
// Class definitions
//////////////////////////////////////////////////////////////
static const Cinfo* zombiePoolCinfo = ZPool::initCinfo();

static const SrcFinfo1< double >* requestSize = 
    dynamic_cast< const SrcFinfo1< double >* >(  
    zombiePoolCinfo->findFinfo( "requestSize" ) );

ZPool::ZPool()
		: solver_( 0 )
{;}

ZPool::~ZPool()
{;}

//////////////////////////////////////////////////////////////
// MsgDest Definitions
//////////////////////////////////////////////////////////////

// vRemesh: All the work is done by the message from the compartment to the
// Stoich. None of the ZPools is remeshed directly. However, their
// DataHandlers need updating.
void ZPool::vRemesh( const Eref& e, const Qinfo* q, 
	double oldvol,
	unsigned int numTotalEntries, unsigned int startEntry, 
	const vector< unsigned int >& localIndices, 
	const vector< double >& vols )
{
	;
	if ( e.index().value() != 0 )
		return;
	Neutral* n = reinterpret_cast< Neutral* >( e.data() );
	if ( vols.size() != e.element()->dataHandler()->localEntries() )
		n->setLastDimension( e, q, vols.size() );
}

//////////////////////////////////////////////////////////////
// Field Definitions
//////////////////////////////////////////////////////////////

void ZPool::vSetN( const Eref& e, const Qinfo* q, double v )
{
	solver_->setN( e, v );
}

double ZPool::vGetN( const Eref& e, const Qinfo* q ) const
{
	return solver_->getN( e );
}

void ZPool::vSetNinit( const Eref& e, const Qinfo* q, double v )
{
	solver_->setNinit( e, v );
}

double ZPool::vGetNinit( const Eref& e, const Qinfo* q ) const
{
	return solver_->getNinit( e );
}

void ZPool::vSetConc( const Eref& e, const Qinfo* q, double conc )
{
	// unsigned int pool = convertIdToPoolIndex( e.id() );
	double n = NA * conc * lookupSizeFromMesh( e, requestSize );
	solver_->setN( e, n );
}

double ZPool::vGetConc( const Eref& e, const Qinfo* q ) const
{
	return solver_->getN( e ) / 
			( NA * lookupSizeFromMesh( e, requestSize ) );
}

void ZPool::vSetConcInit( const Eref& e, const Qinfo* q, double conc )
{
	double n = NA * conc * lookupSizeFromMesh( e, requestSize );
	solver_->setNinit( e, n );
}

double ZPool::vGetConcInit( const Eref& e, const Qinfo* q ) const
{
	return solver_->getNinit( e ) / 
			( NA * lookupSizeFromMesh( e, requestSize ) );
}

void ZPool::vSetDiffConst( const Eref& e, const Qinfo* q, double v )
{
	solver_->setDiffConst( e, v );
}

double ZPool::vGetDiffConst( const Eref& e, const Qinfo* q ) const
{
	return solver_->getDiffConst( e );
}

void ZPool::vSetSize( const Eref& e, const Qinfo* q, double v )
{
	// Illegal operation.
}

double ZPool::vGetSize( const Eref& e, const Qinfo* q ) const
{
	return lookupSizeFromMesh( e, requestSize );
}

void ZPool::vSetSpecies( const Eref& e, const Qinfo* q, unsigned int v )
{
	solver_->setSpecies( e, v );
}

unsigned int ZPool::vGetSpecies( const Eref& e, const Qinfo* q ) const
{
	return solver_->getSpecies( e );
}

//////////////////////////////////////////////////////////////
// Zombie conversion functions.
//////////////////////////////////////////////////////////////
void ZPool::setSolver( Id solver )
{
	assert ( solver != Id() );
	assert( solver.element()->cinfo()->isA( "SolverBase" ) );
	solver_ = reinterpret_cast< SolverBase* >( solver.eref().data() );
}

