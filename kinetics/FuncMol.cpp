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
#include "FuncMol.h"

const Cinfo* FuncMol::initCinfo()
{
		//////////////////////////////////////////////////////////////
		// Field Definitions
		//////////////////////////////////////////////////////////////

		//////////////////////////////////////////////////////////////
		// MsgDest Definitions
		//////////////////////////////////////////////////////////////
		static DestFinfo process( "process",
			"Handles process call",
			new ProcOpFunc< FuncMol >( &FuncMol::process ) );
		static DestFinfo reinit( "reinit",
			"Handles reinit call",
			new ProcOpFunc< FuncMol >( &FuncMol::reinit ) );
		static DestFinfo input( "input",
			"Handles input to control value of n_",
			new OpFunc1< FuncMol, double >( &FuncMol::input ) );

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

	static Finfo* funcMolFinfos[] = {
		&input,				// DestFinfo
		&proc,				// SharedFinfo
	};

	static Cinfo funcMolCinfo (
		"FuncMol",
		Mol::initCinfo(),
		funcMolFinfos,
		sizeof( funcMolFinfos ) / sizeof ( Finfo* ),
		new Dinfo< FuncMol >()
	);

	return &funcMolCinfo;
}

//////////////////////////////////////////////////////////////
// Class definitions
//////////////////////////////////////////////////////////////
static const Cinfo* funcMolCinfo = FuncMol::initCinfo();

FuncMol::FuncMol()
{;}

//////////////////////////////////////////////////////////////
// MsgDest Definitions
//////////////////////////////////////////////////////////////

void FuncMol::process( const Eref& e, ProcPtr p )
{
	Mol::reinit( e, p );
}

void FuncMol::reinit( const Eref& e, ProcPtr p )
{
	Mol::reinit( e, p );
}

void FuncMol::input( double v )
{
	setNinit( v );
}

//////////////////////////////////////////////////////////////
// Field Definitions are all inherited
//////////////////////////////////////////////////////////////

