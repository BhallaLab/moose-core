/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include <queue>
#include "header.h"
#include "Arith.h"

static SrcFinfo1< double > output( 
		"output", 
		"Sends out the computed value"
	);

const Cinfo* Arith::initCinfo()
{
		//////////////////////////////////////////////////////////////
		// Field Definitions
		//////////////////////////////////////////////////////////////
		static ValueFinfo< Arith, string > function(
			"function",
			"Arithmetic function to perform on inputs.",
			&Arith::setFunction,
			&Arith::getFunction
		);
		static ValueFinfo< Arith, double > outputValue(
			"outputValue",
			"Value of output as computed last timestep.",
			&Arith::setOutput,
			&Arith::getOutput
		);
		//////////////////////////////////////////////////////////////
		// MsgDest Definitions
		//////////////////////////////////////////////////////////////
		static DestFinfo arg1( "arg1",
			"Handles argument 1. This just assigns it",
			new OpFunc1< Arith, double >( &Arith::arg1 ) );

		static DestFinfo arg2( "arg2",
			"Handles argument 2. This just assigns it",
			new OpFunc1< Arith, double >( &Arith::arg2 ) );

		static DestFinfo arg3( "arg3",
			"Handles argument 3. This sums in each input, and clears each clock tick.",
			new OpFunc1< Arith, double >( &Arith::arg3 ) );

		static DestFinfo process( "process",
			"Handles process call",
			new ProcOpFunc< Arith >( &Arith::process ) );
		static DestFinfo reinit( "reinit",
			"Handles reinit call",
			new ProcOpFunc< Arith >( &Arith::reinit ) );

		//////////////////////////////////////////////////////////////
		// SharedFinfo Definitions
		//////////////////////////////////////////////////////////////
		static Finfo* procShared[] = {
			&process, &reinit
		};
		static SharedFinfo proc( "proc",
			"Shared message for process and reinit",
			procShared, sizeof( procShared ) / sizeof( const Finfo* )
		);

	static Finfo* arithFinfos[] = {
		&function,	// Value
		&outputValue,	// Value
		&arg1,		// DestFinfo
		&arg2,		// DestFinfo
		&arg3,		// DestFinfo
		&output, 	// SrcFinfo
		&proc		// SharedFinfo
	};

	static Cinfo arithCinfo (
		"Arith",
		Neutral::initCinfo(),
		arithFinfos,
		sizeof( arithFinfos ) / sizeof ( Finfo* ),
		new Dinfo< Arith >()
	);

	return &arithCinfo;
}

static const Cinfo* arithCinfo = Arith::initCinfo();

Arith::Arith()
	: function_( "sum" ), 
	output_( 0.0 ),
	arg1_( 0.0 ), arg2_( 0.0 ), arg3_( 0.0 )
{
	;
}

void Arith::process( const Eref& e, ProcPtr p )
{
	output_ = arg1_ + arg2_ + arg3_; // Doing a hard-coded function.
	output.send( e, p, output_ );
	arg3_ = 0.0;
}

void Arith::reinit( const Eref& e, ProcPtr p )
{
	arg1_ = 0.0;
	arg2_ = 0.0;
	arg3_ = 0.0;
}

void Arith::arg1( const double arg )
{
	arg1_ = arg;
}

void Arith::arg2( const double arg )
{
	arg2_ = arg;
}

void Arith::arg3( const double arg )
{
	arg3_ += arg;
}

void Arith::setFunction( const string v )
{
	function_ = v;
}

string Arith::getFunction() const
{
	return function_;
}

void Arith::setOutput( double v )
{
	output_ = v;
}

double Arith::getOutput() const
{
	return output_;
}
