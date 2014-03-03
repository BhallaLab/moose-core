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
#include "ZombieFuncPool.h"

// Derived from ZombiePool.
const Cinfo* ZombieFuncPool::initCinfo()
{
	static DestFinfo input( "input",
		"Handles input to control value of n_",
		new EpFunc1< ZombieFuncPool, double >( &ZombieFuncPool::input ) );
	
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

void ZombieFuncPool::input( const Eref& e, double v )
{
	ZombiePool::vSetN( e, v );
	ZombiePool::vSetNinit( e, v );
}

