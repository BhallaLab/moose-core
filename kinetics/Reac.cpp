/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
#include "ElementValueFinfo.h"
#include "lookupSizeFromMesh.h"
#include "Reac.h"

#define EPSILON 1e-15

static SrcFinfo2< double, double > *toSub() {
	static SrcFinfo2< double, double > toSub( 
			"toSub", 
			"Sends out increment of molecules on product each timestep"
			);
	return &toSub;
}

static SrcFinfo2< double, double > *toPrd() {
	static SrcFinfo2< double, double > toPrd( 
			"toPrd", 
			"Sends out increment of molecules on product each timestep"
			);
	return &toPrd;
}

const Cinfo* Reac::initCinfo()
{
		//////////////////////////////////////////////////////////////
		// Field Definitions
		//////////////////////////////////////////////////////////////
		static ElementValueFinfo< Reac, double > kf(
			"kf",
			"Forward rate constant, in # units",
			&Reac::setNumKf,
			&Reac::getNumKf
		);

		static ElementValueFinfo< Reac, double > kb(
			"kb",
			"Reverse rate constant, in # units",
			&Reac::setNumKb,
			&Reac::getNumKb
		);

		static ElementValueFinfo< Reac, double > Kf(
			"Kf",
			"Forward rate constant, in concentration units",
			&Reac::setConcKf,
			&Reac::getConcKf
		);

		static ElementValueFinfo< Reac, double > Kb(
			"Kb",
			"Reverse rate constant, in concentration units",
			&Reac::setConcKb,
			&Reac::getConcKb
		);

		//////////////////////////////////////////////////////////////
		// MsgDest Definitions
		//////////////////////////////////////////////////////////////
		static DestFinfo process( "process",
			"Handles process call",
			new ProcOpFunc< Reac >( &Reac::process ) );
		static DestFinfo reinit( "reinit",
			"Handles reinit call",
			new ProcOpFunc< Reac >( &Reac::reinit ) );

		static DestFinfo group( "group",
			"Handle for group msgs. Doesn't do anything",
			new OpFuncDummy() );

		//////////////////////////////////////////////////////////////
		// Shared Msg Definitions
		//////////////////////////////////////////////////////////////

		static DestFinfo subDest( "subDest",
				"Handles # of molecules of substrate",
				new OpFunc1< Reac, double >( &Reac::sub ) );
		static DestFinfo prdDest( "prdDest",
				"Handles # of molecules of product",
				new OpFunc1< Reac, double >( &Reac::prd ) );
		static Finfo* subShared[] = {
			toSub(), &subDest
		};
		static Finfo* prdShared[] = {
			toPrd(), &prdDest
		};
		static SharedFinfo sub( "sub",
			"Connects to substrate pool",
			subShared, sizeof( subShared ) / sizeof( const Finfo* )
		);
		static SharedFinfo prd( "prd",
			"Connects to substrate pool",
			prdShared, sizeof( prdShared ) / sizeof( const Finfo* )
		);
		static Finfo* procShared[] = {
			&process, &reinit
		};
		static SharedFinfo proc( "proc",
			"Shared message for process and reinit",
			procShared, sizeof( procShared ) / sizeof( const Finfo* )
		);


	static Finfo* reacFinfos[] = {
		&kf,	// Value
		&kb,	// Value
		&Kf,	// Value
		&Kb,	// Value
		&sub,				// SharedFinfo
		&prd,				// SharedFinfo
		&proc,				// SharedFinfo
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
	: kf_( 0.1 ), kb_( 0.2 ), sub_( 0.0 ), prd_( 0.0 )
{
	;
}

Reac::Reac( double kf, double kb )
	: kf_( kf ), kb_( kb ), sub_( 0.0 ), prd_( 0.0 )
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

void Reac::process( const Eref& e, ProcPtr p )
{
	toPrd()->send( e, p->threadIndexInGroup, sub_, prd_ );
	toSub()->send( e, p->threadIndexInGroup, prd_, sub_ );
	
	sub_ = kf_;
	prd_ = kb_;
}

void Reac::reinit( const Eref& e, ProcPtr p )
{
	sub_ = kf_ = concKf_ *
		convertConcToNumRateUsingMesh( e, toSub(), 0, CONC_UNIT_CONV, 0 );
	prd_ = kb_ = concKb_ * 
		convertConcToNumRateUsingMesh( e, toPrd(), 0, CONC_UNIT_CONV, 0 );
}

//////////////////////////////////////////////////////////////
// Field Definitions
//////////////////////////////////////////////////////////////

void Reac::setNumKf( const Eref& e, const Qinfo* q, double v )
{
	sub_ = kf_ = v;
	double volScale = 
		convertConcToNumRateUsingMesh( e, toSub(), 0, CONC_UNIT_CONV, 0 );
	concKf_ = kf_ / volScale;
}

double Reac::getNumKf( const Eref& e, const Qinfo* q) const
{
	double kf = concKf_ / 
		convertConcToNumRateUsingMesh( e, toSub(), 0, CONC_UNIT_CONV, 0 );
	return kf;
}

void Reac::setNumKb( const Eref& e, const Qinfo* q, double v )
{
	prd_ = kb_ = v;
	double volScale = 
		convertConcToNumRateUsingMesh( e, toPrd(), 0, CONC_UNIT_CONV, 0 );
	concKb_ = kb_ / volScale;
}

double Reac::getNumKb( const Eref& e, const Qinfo* q ) const
{
	double kb = concKb_ / 
		convertConcToNumRateUsingMesh( e, toPrd(), 0, CONC_UNIT_CONV, 0 );
	return kb;
}

void Reac::setConcKf( const Eref& e, const Qinfo* q, double v )
{
	concKf_ = v;
	sub_ = kf_ = v * 
		convertConcToNumRateUsingMesh( e, toSub(), 0, CONC_UNIT_CONV, 0 );
}

double Reac::getConcKf( const Eref& e, const Qinfo* q ) const
{
	return concKf_;
}

void Reac::setConcKb( const Eref& e, const Qinfo* q, double v )
{
	concKb_ = v;
	prd_ = kb_ = 
		v * convertConcToNumRateUsingMesh( e, toPrd(), 0, CONC_UNIT_CONV, 0);
}

double Reac::getConcKb( const Eref& e, const Qinfo* q ) const
{
	return concKb_;
}
