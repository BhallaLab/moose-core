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
#include "Pool.h"
#include "FuncPool.h"

const Cinfo* FuncPool::initCinfo()
{
		//////////////////////////////////////////////////////////////
		// Field Definitions
		//////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////
		// MsgDest Definitions
		//////////////////////////////////////////////////////////////
		static DestFinfo input( "input",
			"Handles input to control value of n_",
			new EpFunc1< FuncPool, double >( &FuncPool::input ) );

		//////////////////////////////////////////////////////////////
		// SharedMsg Definitions
		//////////////////////////////////////////////////////////////

	static Finfo* funcPoolFinfos[] = {
		&input,				// DestFinfo
	};

	static Dinfo< FuncPool > dinfo;
	static Cinfo funcPoolCinfo (
		"FuncPool",
		Pool::initCinfo(),
		funcPoolFinfos,
		sizeof( funcPoolFinfos ) / sizeof ( Finfo* ),
		&dinfo
	);

	return &funcPoolCinfo;
}

//////////////////////////////////////////////////////////////
// Class definitions
//////////////////////////////////////////////////////////////
static const Cinfo* funcPoolCinfo = FuncPool::initCinfo();

FuncPool::FuncPool()
{;}

FuncPool::~FuncPool()
{;}

//////////////////////////////////////////////////////////////
// MsgDest Definitions
//////////////////////////////////////////////////////////////

void FuncPool::vProcess( const Eref& e, ProcPtr p )
{
	;
}

void FuncPool::vReinit( const Eref& e, ProcPtr p )
{
	Pool::vReinit( e, p );
}

void FuncPool::input( const Eref& e, double v )
{
	Pool::vSetN( e, v );
	Pool::vSetNinit( e, v );
}

//////////////////////////////////////////////////////////////
// Field Definitions are all inherited
//////////////////////////////////////////////////////////////

