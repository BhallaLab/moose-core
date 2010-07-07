/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
#include "Mol.h"
#include "BufMol.h"

#define EPSILON 1e-15

const Cinfo* BufMol::initCinfo()
{
		//////////////////////////////////////////////////////////////
		// Field Definitions
		//////////////////////////////////////////////////////////////

		//////////////////////////////////////////////////////////////
		// MsgDest Definitions
		//////////////////////////////////////////////////////////////
		static DestFinfo process( "process",
			"Handles process call",
			new ProcOpFunc< BufMol >( &BufMol::process ) );
		static DestFinfo reinit( "reinit",
			"Handles reinit call",
			new ProcOpFunc< BufMol >( &BufMol::reinit ) );

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

	static Finfo* bufMolFinfos[] = {
		&proc,				// SharedFinfo
	};

	static Cinfo bufMolCinfo (
		"BufMol",
		Mol::initCinfo(),
		bufMolFinfos,
		sizeof( bufMolFinfos ) / sizeof ( Finfo* ),
		new Dinfo< BufMol >()
	);

	return &bufMolCinfo;
}

//////////////////////////////////////////////////////////////
// Class definitions
//////////////////////////////////////////////////////////////
static const Cinfo* bufMolCinfo = BufMol::initCinfo();

BufMol::BufMol()
{;}

BufMol::BufMol( double nInit)
	: Mol( nInit )
{;}

//////////////////////////////////////////////////////////////
// MsgDest Definitions
//////////////////////////////////////////////////////////////

void BufMol::process( const Eref& e, ProcPtr p )
{
	Mol::reinit( e, p );
}

void BufMol::reinit( const Eref& e, ProcPtr p )
{
	Mol::reinit( e, p );
}


//////////////////////////////////////////////////////////////
// Field Definitions
//////////////////////////////////////////////////////////////

