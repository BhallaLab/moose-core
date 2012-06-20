/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/
#include "StoichHeaders.h"
#include "ElementValueFinfo.h"
#include "DataHandlerWrapper.h"

#include "Pool.h"
#include "BufPool.h"
#include "ZombiePool.h"
#include "ZombieBufPool.h"

// Entirely derived from ZombiePool. Only the zombification routines differ.
const Cinfo* ZombieBufPool::initCinfo()
{
		//////////////////////////////////////////////////////////////
		// Field Definitions: use virtual functions to deal with, the
		// moose definitions are inherited.
		//////////////////////////////////////////////////////////////
	static Cinfo zombieBufPoolCinfo (
		"ZombieBufPool",
		ZombiePool::initCinfo(),
		0,
		0,
		new Dinfo< ZombieBufPool >()
	);

	return &zombieBufPoolCinfo;
}

//////////////////////////////////////////////////////////////
// Class definitions
//////////////////////////////////////////////////////////////
static const Cinfo* zombieBufPoolCinfo = ZombieBufPool::initCinfo();

static const SrcFinfo1< double >* requestSize =
	dynamic_cast< const SrcFinfo1< double >* >(
	zombieBufPoolCinfo->findFinfo( "requestSize" ) );


ZombieBufPool::ZombieBufPool()
{;}

ZombieBufPool::~ZombieBufPool()
{;}

//////////////////////////////////////////////////////////////
// Field functions
//////////////////////////////////////////////////////////////

void ZombieBufPool::vSetN( const Eref& e, const Qinfo* q, double v )
{
	stoich_->innerSetN( e.index().value(), e.id(), v );
	stoich_->innerSetNinit( e.index().value(), e.id(), v );
}

void ZombieBufPool::vSetNinit( const Eref& e, const Qinfo* q, double v )
{
	vSetN( e, q, v );
}

void ZombieBufPool::vSetConc( const Eref& e, const Qinfo* q, double conc )
{
	double n = NA * conc * lookupSizeFromMesh( e, requestSize );
	vSetN( e, q, n );
}

void ZombieBufPool::vSetConcInit( const Eref& e, const Qinfo* q, double conc )
{
	vSetConc( e, q, conc );
}

//////////////////////////////////////////////////////////////
// Zombie conversion functions.
//////////////////////////////////////////////////////////////

// static func
void ZombieBufPool::zombify( Element* solver, Element* orig )
{
	DataHandler* dh = orig->dataHandler()->copyUsingNewDinfo(
		ZombieBufPool::initCinfo()->dinfo() );
	Element temp( orig->id(), zombieBufPoolCinfo, dh );
	Eref zer( &temp, 0 );
	Eref oer( orig, 0 );

	ZombieBufPool* z = reinterpret_cast< ZombieBufPool* >( zer.data() );
	PoolBase* m = reinterpret_cast< PoolBase* >( oer.data() );

	z->stoich_ = reinterpret_cast< Stoich* >( solver->dataHandler()->data( 0 ) );
	z->vSetNinit( zer, 0, m->getNinit( oer, 0 ) );
	z->vSetDiffConst( zer, 0, m->getDiffConst( oer, 0 ) );
	orig->zombieSwap( zombieBufPoolCinfo, dh );
}

// Static func
void ZombieBufPool::unzombify( Element* zombie )
{
	Element temp( zombie->id(), zombie->cinfo(), zombie->dataHandler() );
	Eref zer( &temp, 0 );
	Eref oer( zombie, 0 );

	ZombieBufPool* z = reinterpret_cast< ZombieBufPool* >( zer.data() );

	// Here I am unsure how to recreate the correct kind of data handler
	// for the original. Do later.
	DataHandler* dh = 0;

	zombie->zombieSwap( BufPool::initCinfo(), dh );

	BufPool* m = reinterpret_cast< BufPool* >( oer.data() );

	m->setN( oer, 0, z->vGetN( zer, 0 ) );
	m->setNinit( oer, 0, z->vGetNinit( zer, 0 ) );
}
