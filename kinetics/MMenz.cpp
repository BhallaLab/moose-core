/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
#include "MMenz.h"

#define EPSILON 1e-15

static SrcFinfo2< double, double > toSub( 
		"toSub", 
		"Sends out increment of molecules on product each timestep"
	);

static SrcFinfo2< double, double > toPrd( 
		"toPrd", 
		"Sends out increment of molecules on product each timestep"
	);

static DestFinfo sub( "subDest",
		"Handles # of molecules of substrate",
		new OpFunc1< MMenz, double >( &MMenz::sub ) );

static DestFinfo enzDest( "enz",
		"Handles # of molecules of MMenzyme",
		new OpFunc1< MMenz, double >( &MMenz::enz ) );

static DestFinfo prd( "prdDest",
		"Handles # of molecules of product. Dummy.",
		new OpFunc1< MMenz, double >( &MMenz::prd ) );
	
static Finfo* subShared[] = {
	&toSub, &sub
};

static Finfo* prdShared[] = {
	&toPrd, &prd
};

const Cinfo* MMenz::initCinfo()
{
		//////////////////////////////////////////////////////////////
		// Field Definitions
		//////////////////////////////////////////////////////////////
		static ValueFinfo< MMenz, double > Km(
			"Km",
			"Michaelis-Menten constant",
			&MMenz::setKm,
			&MMenz::getKm
		);

		static ValueFinfo< MMenz, double > kcat(
			"kcat",
			"Forward rate constant for enzyme",
			&MMenz::setKcat,
			&MMenz::getKcat
		);

		//////////////////////////////////////////////////////////////
		// MsgDest Definitions
		//////////////////////////////////////////////////////////////
		static DestFinfo process( "process",
			"Handles process call",
			new ProcOpFunc< MMenz >( &MMenz::process ) );
		static DestFinfo reinit( "reinit",
			"Handles reinit call",
			new ProcOpFunc< MMenz >( &MMenz::reinit ) );

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
			"Connects to product molecule",
			prdShared, sizeof( prdShared ) / sizeof( const Finfo* )
		);
		static Finfo* procShared[] = {
			&process, &reinit
		};
		static SharedFinfo proc( "proc",
			"Shared message for process and reinit",
			procShared, sizeof( procShared ) / sizeof( const Finfo* )
		);

	static Finfo* mmEnzFinfos[] = {
		&Km,	// Value
		&kcat,	// Value
		&enzDest,				// DestFinfo
		&sub,				// SharedFinfo
		&prd,				// SharedFinfo
		&proc,				// SharedFinfo
	};

	static Cinfo mmEnzCinfo (
		"MMenz",
		Neutral::initCinfo(),
		mmEnzFinfos,
		sizeof( mmEnzFinfos ) / sizeof ( Finfo* ),
		new Dinfo< MMenz >()
	);

	return &mmEnzCinfo;
}

 static const Cinfo* mmEnzCinfo = MMenz::initCinfo();

//////////////////////////////////////////////////////////////
// MMenz internal functions
//////////////////////////////////////////////////////////////


MMenz::MMenz( )
	: Km_( 5 ), kcat_( 0.1 ), sub_( 0.0 ), enz_( 0.0 )
{
	;
}

//////////////////////////////////////////////////////////////
// MsgDest Definitions
//////////////////////////////////////////////////////////////

void MMenz::sub( double n )
{
	sub_ *= n;
}

void MMenz::prd( double n ) // dummy
{
	;
}

void MMenz::enz( double n )
{
	enz_ = n;
}

void MMenz::process( const Eref& e, ProcPtr p )
{
	double rate = kcat_ * enz_ * sub_ / ( Km_ + sub_ );
	toSub.send( e, p, 0, rate );
	toPrd.send( e, p, rate, 0 );
	
	sub_ = 1.0;
}

void MMenz::reinit( const Eref& e, ProcPtr p )
{
	sub_ = 1.0;
	enz_ = 0.0;
}

//////////////////////////////////////////////////////////////
// Field Definitions
//////////////////////////////////////////////////////////////

void MMenz::setKm( double v )
{
	Km_ = v;
}

double MMenz::getKm() const
{
	return Km_;
}

void MMenz::setKcat( double v )
{
	kcat_ = v;
}

double MMenz::getKcat() const
{
	return kcat_;
}
