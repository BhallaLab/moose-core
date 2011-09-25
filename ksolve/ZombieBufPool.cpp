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

ZombieBufPool::ZombieBufPool()
{;}


//////////////////////////////////////////////////////////////
// Zombie conversion functions.
//////////////////////////////////////////////////////////////

// static func
void ZombieBufPool::zombify( Element* solver, Element* orig )
{
	Element temp( orig->id(), zombieBufPoolCinfo, solver->dataHandler() );
	Eref zer( &temp, 0 );
	Eref oer( orig, 0 );

	ZombieBufPool* z = reinterpret_cast< ZombieBufPool* >( zer.data() );
	BufPool* m = reinterpret_cast< BufPool* >( oer.data() );

	z->setN( zer, 0, m->getN() );
	z->setNinit( zer, 0, m->getNinit() );
	z->setDiffConst( zer, 0, m->getDiffConst() );
	DataHandler* dh = new DataHandlerWrapper( solver->dataHandler() );
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

	m->setN( z->getN( zer, 0 ) );
	m->setNinit( z->getNinit( zer, 0 ) );
}
