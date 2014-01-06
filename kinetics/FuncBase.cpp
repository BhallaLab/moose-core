/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2012 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
#include "FuncTerm.h"
#include "FuncBase.h"

static SrcFinfo1< double >* output()
{
	static SrcFinfo1< double > output(
			"output", 
			"Sends out sum on each timestep"
	);
	return &output;
}

const Cinfo* FuncBase::initCinfo()
{
		//////////////////////////////////////////////////////////////
		// Field Definitions
		//////////////////////////////////////////////////////////////
		static ReadOnlyValueFinfo< FuncBase, double > result(
			"result",
			"Outcome of function computation",
			&FuncBase::getResult
		);

		//////////////////////////////////////////////////////////////
		// MsgDest Definitions
		//////////////////////////////////////////////////////////////
		static DestFinfo process( "process",
			"Handles process call",
			new ProcOpFunc< FuncBase >( &FuncBase::process ) );
		static DestFinfo reinit( "reinit",
			"Handles reinit call",
			new ProcOpFunc< FuncBase >( &FuncBase::reinit ) );
		static DestFinfo input( "input",
			"Handles input values."
			" This generic message works only in cases where the inputs "
			" are commutative, so ordering does not matter. "
			" In due course will implement a synapse type extendable, "
			" identified system of inputs so that arbitrary numbers of "
			" inputs can be unambiguaously defined. ",
			new OpFunc1< FuncBase, double >( &FuncBase::input ) );

		//////////////////////////////////////////////////////////////
		// SrcFinfo Definitions
		//////////////////////////////////////////////////////////////
		//See above.

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

	static Finfo* funcBaseFinfos[] = {
		&result,	// Value
		&input,				// DestFinfo
		output(),			// SrcFinfo
		&proc,				// SharedFinfo
	};

	static ZeroSizeDinfo< int > dinfo;
	static Cinfo funcBaseCinfo (
		"FuncBase",
		Neutral::initCinfo(),
		funcBaseFinfos,
		sizeof( funcBaseFinfos ) / sizeof ( Finfo* ),
		&dinfo
	);

	return &funcBaseCinfo;
}

//////////////////////////////////////////////////////////////
// Class definitions
//////////////////////////////////////////////////////////////
static const Cinfo* funcBaseCinfo = FuncBase::initCinfo();

FuncBase::FuncBase()
	: result_( 0.0 )
{;}

FuncBase::~FuncBase()
{;}

//////////////////////////////////////////////////////////////
// MsgDest Definitions
//////////////////////////////////////////////////////////////

void FuncBase::process( const Eref& e, ProcPtr p )
{
	vProcess( e, p );
}

void FuncBase::reinit( const Eref& e, ProcPtr p )
{
	vReinit( e, p );
}

void FuncBase::input( double v )
{
	vInput( v );
}

//////////////////////////////////////////////////////////////
// Field Definitions
//////////////////////////////////////////////////////////////

double FuncBase::getResult() const
{
	return result_;
}

