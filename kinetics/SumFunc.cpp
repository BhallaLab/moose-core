/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
#include "FuncTerm.h"
#include "SumTotalTerm.h"
#include "FuncBase.h"
#include "SumFunc.h"

const Cinfo* SumFunc::initCinfo()
{
	static string doc[] =
	{
			"Name", "SumFunc",
			"Author", "Upi Bhalla",
			"Description",
			"SumFunc object. Adds up all inputs",
	};

	static Dinfo< SumFunc > dinfo;
	static Cinfo sumFuncCinfo (
		"SumFunc",
		FuncBase::initCinfo(),
		0, 0,
		&dinfo,
		doc,
		sizeof( doc ) / sizeof( string )
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
	: FuncBase()
{;}

SumFunc::~SumFunc()
{;}

//////////////////////////////////////////////////////////////
// MsgDest Definitions
//////////////////////////////////////////////////////////////

void SumFunc::vProcess( const Eref& e, ProcPtr p )
{
	output.send( e, result_ );
	result_ = 0;
}

void SumFunc::vReinit( const Eref& e, ProcPtr p )
{
	vProcess( e, p );
}

void SumFunc::vInput( double v )
{
	result_ += v;
}

//////////////////////////////////////////////////////////////

FuncTerm* SumFunc::func()
{
	return &st_;
}
