/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/
#include "StoichHeaders.h"
#include "ZPool.h"
#include "ZFuncPool.h"
// #include "ZSumFunc.h"

// Derived from ZPool.
const Cinfo* ZFuncPool::initCinfo()
{
	static DestFinfo input( "input",
		"Handles input to control value of n_",
		new OpFunc1< ZFuncPool, double >( &ZFuncPool::input ) );
	
	static Finfo* zombieFuncPoolFinfos[] = {
		&input,             // DestFinfo
	};

	static Cinfo zombieFuncPoolCinfo (
		"ZFuncPool",
		ZPool::initCinfo(),
		zombieFuncPoolFinfos,
		sizeof( zombieFuncPoolFinfos ) / sizeof( const Finfo* ),
		new Dinfo< ZFuncPool >()
	);

	return &zombieFuncPoolCinfo;
}

//////////////////////////////////////////////////////////////
// Class definitions
//////////////////////////////////////////////////////////////
static const Cinfo* zombieFuncPoolCinfo = ZFuncPool::initCinfo();

ZFuncPool::ZFuncPool()
{;}

void ZFuncPool::input( double v )
{;}

