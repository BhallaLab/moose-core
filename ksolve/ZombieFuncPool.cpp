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
#include "FuncPool.h"
#include "ZombiePool.h"
#include "ZombieFuncPool.h"
#include "ZombieSumFunc.h"

// Derived from ZombiePool.
const Cinfo* ZombieFuncPool::initCinfo()
{
	static DestFinfo input( "input",
		"Handles input to control value of n_",
		new OpFunc1< ZombieFuncPool, double >( &ZombieFuncPool::input ) );
	
	static Finfo* zombieFuncPoolFinfos[] = {
		&input,             // DestFinfo
	};

	static Cinfo zombieFuncPoolCinfo (
		"ZombieFuncPool",
		ZombiePool::initCinfo(),
		zombieFuncPoolFinfos,
		sizeof( zombieFuncPoolFinfos ) / sizeof( const Finfo* ),
		new Dinfo< ZombieFuncPool >()
	);

	return &zombieFuncPoolCinfo;
}

//////////////////////////////////////////////////////////////
// Class definitions
//////////////////////////////////////////////////////////////
static const Cinfo* zombieFuncPoolCinfo = ZombieFuncPool::initCinfo();

ZombieFuncPool::ZombieFuncPool()
{;}

void ZombieFuncPool::input( double v )
{;}

//////////////////////////////////////////////////////////////
// Zombie conversion functions.
//////////////////////////////////////////////////////////////

// static func
// This is more involved as it also has to zombify the Func.
void ZombieFuncPool::zombify( Element* solver, Element* orig )
{
	Element temp( orig->id(), zombieFuncPoolCinfo, solver->dataHandler() );
	Eref zer( &temp, 0 );
	Eref oer( orig, 0 );

	ZombieFuncPool* z = reinterpret_cast< ZombieFuncPool* >( zer.data() );
	FuncPool* m = reinterpret_cast< FuncPool* >( oer.data() );

	z->setN( zer, 0, m->getN() );
	z->setNinit( zer, 0, m->getNinit( oer, 0 ) );
	DataHandler* dh = new DataHandlerWrapper( solver->dataHandler(),
		orig->dataHandler() );
	orig->zombieSwap( zombieFuncPoolCinfo, dh );

	// Later change name just to 'func'
	Id funcId = Neutral::child( oer, "sumFunc" );
	if ( funcId != Id() ) {
		if ( funcId()->cinfo()->isA( "SumFunc" ) )
			ZombieSumFunc::zombify( solver, funcId(), orig->id() );
			// The additional Id argument helps the system to look up
			// what molecule is involved. Could of course get it as target.
		/*
		else if ( funcId()->cinfo().isA( "MathFunc" ) )
			ZombieMathFunc::zombify( solver, funcId() );
		else if ( funcId()->cinfo().isA( "Table" ) )
			ZombieTable::zombify( solver, funcId() );
		*/
	}
}

// Static func
void ZombieFuncPool::unzombify( Element* zombie )
{
	Element temp( zombie->id(), zombie->cinfo(), zombie->dataHandler() );
	Eref zer( &temp, 0 );
	Eref oer( zombie, 0 );

	ZombieFuncPool* z = reinterpret_cast< ZombieFuncPool* >( zer.data() );

	// Here I am unsure how to recreate the correct kind of data handler
	// for the original. Do later.
	DataHandler* dh = 0;

	zombie->zombieSwap( FuncPool::initCinfo(), dh );

	FuncPool* m = reinterpret_cast< FuncPool* >( oer.data() );

	m->setN( z->getN( zer, 0 ) );
	m->setNinit( oer, 0, z->getNinit( zer, 0 ) );
}
