/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
#include "Reac.h"

#define EPSILON 1e-15

static SrcFinfo2< double, double > reac( 
		"reac", 
		"Sends out increment of molecules on product each timestep"
	);

static DestFinfo sub( "subDest",
		"Handles # of molecules of substrate",
		new OpFunc1< Reac, double >( &Reac::sub ) );

static DestFinfo prd( "prdDest",
		"Handles # of molecules of product",
		new OpFunc1< Reac, double >( &Reac::sub ) );
	
static Finfo* subShared[] = {
	&reac, &sub
};

static Finfo* prdShared[] = {
	&reac, &prd
};

const Cinfo* Reac::initCinfo()
{
		//////////////////////////////////////////////////////////////
		// Field Definitions
		//////////////////////////////////////////////////////////////
		static ValueFinfo< Reac, double > kf(
			"kf",
			"Forward rate constant",
			&Reac::setKf,
			&Reac::getKf
		);

		static ValueFinfo< Reac, double > kb(
			"kb",
			"Forward rate constant",
			&Reac::setKb,
			&Reac::getKb
		);

		//////////////////////////////////////////////////////////////
		// MsgDest Definitions
		//////////////////////////////////////////////////////////////
		static DestFinfo process( "process",
			"Handles process call",
			new EpFunc1< Reac, ProcPtr >( &Reac::eprocess ) );

		static DestFinfo group( "group",
			"Handle for group msgs. Doesn't do anything",
			new OpFuncDummy() );

		//////////////////////////////////////////////////////////////
		// Shared Msg Definitions
		//////////////////////////////////////////////////////////////
		static SharedFinfo sub( "sub",
			"Connects to substrate molecule",
			subShared, sizeof( subShared ) / sizeof( const Finfo* )
		);
		static SharedFinfo prd( "prd",
			"Connects to substrate molecule",
			prdShared, sizeof( prdShared ) / sizeof( const Finfo* )
		);

	static Finfo* reacFinfos[] = {
		&kf,	// Value
		&kb,	// Value
		&process,			// DestFinfo
		&sub,				// SharedFinfo
		&prd,				// SharedFinfo
	};

	static Cinfo reacCinfo (
		"Reac",
		Neutral::initCinfo(),
		reacFinfos,
		sizeof( reacFinfos ) / sizeof ( Finfo* ),
		new Dinfo< Reac >()
	);

	return &reacCinfo;
}

 static const Cinfo* reacCinfo = Reac::initCinfo();

//////////////////////////////////////////////////////////////
// Reac internal functions
//////////////////////////////////////////////////////////////


Reac::Reac( )
	: kf_( 0.1 ), kb_( 0.2 )
{
	;
}

Reac::Reac( double kf, double kb )
	: kf_( kf ), kb_( kb )
{
	;
}

//////////////////////////////////////////////////////////////
// MsgDest Definitions
//////////////////////////////////////////////////////////////

void Reac::sub( double v )
{
	sub_ *= v;
}

void Reac::prd( double v )
{
	prd_ *= v;
}

void Reac::eprocess( Eref e, const Qinfo* q, ProcInfo* p )
{
	process( p, e );
}

void Reac::process( const ProcInfo* p, const Eref& e )
{
	reac.send( e, p, sub_, prd_ );
	reac.send( e, p, prd_, sub_ );
	
	sub_ = kf_;
	prd_ = kb_;
}

void Reac::reinit( const Eref& e, const Qinfo*q, ProcInfo* p )
{
	;
}

//////////////////////////////////////////////////////////////
// Field Definitions
//////////////////////////////////////////////////////////////

void Reac::setKf( double v )
{
	kf_ = v;
}

double Reac::getKf() const
{
	return kf_;
}

void Reac::setKb( double v )
{
	kb_ = v;
}

double Reac::getKb() const
{
	return kb_;
}

