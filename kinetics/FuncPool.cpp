/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
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
		static DestFinfo process( "process",
			"Handles process call",
			new ProcOpFunc< FuncPool >( &FuncPool::process ) );
		static DestFinfo reinit( "reinit",
			"Handles reinit call",
			new ProcOpFunc< FuncPool >( &FuncPool::reinit ) );
		static DestFinfo input( "input",
			"Handles input to control value of n_",
			new OpFunc1< FuncPool, double >( &FuncPool::input ) );

		//////////////////////////////////////////////////////////////
		// SharedMsg Definitions
		//////////////////////////////////////////////////////////////
		static Finfo* procShared[] = {
			&process, &reinit
		};
		static SharedFinfo proc( "proc",
			"Shared message for process and reinit",
			procShared, sizeof( procShared ) / sizeof( const Finfo* )
		);

	static Finfo* funcPoolFinfos[] = {
		&input,				// DestFinfo
		&proc,				// SharedFinfo
	};

	static Cinfo funcPoolCinfo (
		"FuncPool",
		Pool::initCinfo(),
		funcPoolFinfos,
		sizeof( funcPoolFinfos ) / sizeof ( Finfo* ),
		new Dinfo< FuncPool >()
	);

	return &funcPoolCinfo;
}

//////////////////////////////////////////////////////////////
// Class definitions
//////////////////////////////////////////////////////////////
static const Cinfo* funcPoolCinfo = FuncPool::initCinfo();

FuncPool::FuncPool()
{;}

//////////////////////////////////////////////////////////////
// MsgDest Definitions
//////////////////////////////////////////////////////////////

void FuncPool::process( const Eref& e, ProcPtr p )
{
	Pool::reinit( e, p );
}

void FuncPool::reinit( const Eref& e, ProcPtr p )
{
	Pool::reinit( e, p );
}

void FuncPool::input( double v )
{
	setConcInit( v );
}

//////////////////////////////////////////////////////////////
// Field Definitions are all inherited
//////////////////////////////////////////////////////////////

