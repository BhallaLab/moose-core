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
#include "BufPool.h"

#define EPSILON 1e-15

const Cinfo* BufPool::initCinfo()
{
		//////////////////////////////////////////////////////////////
		// Field Definitions
		//////////////////////////////////////////////////////////////

		//////////////////////////////////////////////////////////////
		// MsgDest Definitions
		//////////////////////////////////////////////////////////////
		static DestFinfo process( "process",
			"Handles process call",
			new ProcOpFunc< BufPool >( &BufPool::process ) );
		static DestFinfo reinit( "reinit",
			"Handles reinit call",
			new ProcOpFunc< BufPool >( &BufPool::reinit ) );

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

	static Finfo* bufPoolFinfos[] = {
		&proc,				// SharedFinfo
	};

	static Cinfo bufPoolCinfo (
		"BufPool",
		Pool::initCinfo(),
		bufPoolFinfos,
		sizeof( bufPoolFinfos ) / sizeof ( Finfo* ),
		new Dinfo< BufPool >()
	);

	return &bufPoolCinfo;
}

//////////////////////////////////////////////////////////////
// Class definitions
//////////////////////////////////////////////////////////////
static const Cinfo* bufPoolCinfo = BufPool::initCinfo();

BufPool::BufPool()
{;}

BufPool::BufPool( double nInit)
	: Pool( nInit )
{;}

//////////////////////////////////////////////////////////////
// MsgDest Definitions
//////////////////////////////////////////////////////////////

void BufPool::process( const Eref& e, ProcPtr p )
{
	Pool::reinit( e, p );
}

void BufPool::reinit( const Eref& e, ProcPtr p )
{
	Pool::reinit( e, p );
}


//////////////////////////////////////////////////////////////
// Field Definitions
//////////////////////////////////////////////////////////////

