/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
#include "SumFunc.h"

const Cinfo* SumFunc::initCinfo()
{
		//////////////////////////////////////////////////////////////
		// Field Definitions
		//////////////////////////////////////////////////////////////
		static ReadOnlyValueFinfo< SumFunc, double > result(
			"result",
			"outcome of summation",
			&SumFunc::getResult
		);

		//////////////////////////////////////////////////////////////
		// MsgDest Definitions
		//////////////////////////////////////////////////////////////
		static DestFinfo process( "process",
			"Handles process call",
			new ProcOpFunc< SumFunc >( &SumFunc::process ) );
		static DestFinfo reinit( "reinit",
			"Handles reinit call",
			new ProcOpFunc< SumFunc >( &SumFunc::reinit ) );
		static DestFinfo input( "input",
			"Handles input values",
			new OpFunc1< SumFunc, double >( &SumFunc::input ) );

		//////////////////////////////////////////////////////////////
		// SrcFinfo Definitions
		//////////////////////////////////////////////////////////////

		static SrcFinfo1< double > output(
				"output", 
				"Sends out sum on each timestep"
		);

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

	static Finfo* sumFuncFinfos[] = {
		&result,	// Value
		&input,				// DestFinfo
		&output,			// SrcFinfo
		&proc,				// SharedFinfo
	};

	static Cinfo sumFuncCinfo (
		"SumFunc",
		Neutral::initCinfo(),
		sumFuncFinfos,
		sizeof( sumFuncFinfos ) / sizeof ( Finfo* ),
		new Dinfo< SumFunc >()
	);

	return &sumFuncCinfo;
}

//////////////////////////////////////////////////////////////
// Class definitions
//////////////////////////////////////////////////////////////
static const Cinfo* sumFuncCinfo = SumFunc::initCinfo();
const SrcFinfo1< double >& output = 
	*dynamic_cast< const SrcFinfo1< double >* >( 
	sumFuncCinfo->findFinfo( "output" ) );


SumFunc::SumFunc()
	: result_( 0.0 )
{;}

//////////////////////////////////////////////////////////////
// MsgDest Definitions
//////////////////////////////////////////////////////////////

void SumFunc::process( const Eref& e, ProcPtr p )
{
	output.send( e, p->threadIndexInGroup, result_ );
	result_ = 0.0;
}

void SumFunc::reinit( const Eref& e, ProcPtr p )
{
	process( e, p );
}

void SumFunc::input( double v )
{
	result_ += v;
}

//////////////////////////////////////////////////////////////
// Field Definitions
//////////////////////////////////////////////////////////////

double SumFunc::getResult() const
{
	return result_;
}
