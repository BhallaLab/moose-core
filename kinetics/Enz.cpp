/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
#include "Enz.h"

#define EPSILON 1e-15

static SrcFinfo2< double, double > toSub( 
		"toSub", 
		"Sends out increment of molecules on product each timestep"
	);

static SrcFinfo2< double, double > toPrd( 
		"toPrd", 
		"Sends out increment of molecules on product each timestep"
	);
	
static SrcFinfo2< double, double > toEnz( 
		"toEnz", 
		"Sends out increment of molecules on product each timestep"
	);
static SrcFinfo2< double, double > toCplx( 
		"toCplx", 
		"Sends out increment of molecules on product each timestep"
	);

static DestFinfo sub( "subDest",
		"Handles # of molecules of substrate",
		new OpFunc1< Enz, double >( &Enz::sub ) );

static DestFinfo enz( "enzDest",
		"Handles # of molecules of Enzyme",
		new OpFunc1< Enz, double >( &Enz::enz ) );

static DestFinfo prd( "prdDest",
		"Handles # of molecules of product. Dummy.",
		new OpFunc1< Enz, double >( &Enz::prd ) );

static DestFinfo cplx( "prdDest",
		"Handles # of molecules of enz-sub complex",
		new OpFunc1< Enz, double >( &Enz::cplx ) );
	
static Finfo* subShared[] = {
	&toSub, &sub
};

static Finfo* enzShared[] = {
	&toEnz, &enz
};

static Finfo* prdShared[] = {
	&toPrd, &prd
};

static Finfo* cplxShared[] = {
	&toCplx, &cplx
};

const Cinfo* Enz::initCinfo()
{
		//////////////////////////////////////////////////////////////
		// Field Definitions
		//////////////////////////////////////////////////////////////
		static ValueFinfo< Enz, double > k1(
			"k1",
			"Forward rate constant",
			&Enz::setK1,
			&Enz::getK1
		);

		static ValueFinfo< Enz, double > k2(
			"k2",
			"Forward rate constant",
			&Enz::setK2,
			&Enz::getK2
		);

		static ValueFinfo< Enz, double > k3(
			"k3",
			"Forward rate constant",
			&Enz::setK3,
			&Enz::getK3
		);

		//////////////////////////////////////////////////////////////
		// MsgDest Definitions
		//////////////////////////////////////////////////////////////
		static DestFinfo process( "process",
			"Handles process call",
			new EpFunc1< Enz, ProcPtr >( &Enz::eprocess ) );

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
		static SharedFinfo enz( "enz",
			"Connects to enzyme molecule",
			enzShared, sizeof( enzShared ) / sizeof( const Finfo* )
		);
		static SharedFinfo cplx( "cplx",
			"Connects to enz-sub complex molecule",
			cplxShared, sizeof( cplxShared ) / sizeof( const Finfo* )
		);

	static Finfo* enzFinfos[] = {
		&k1,	// Value
		&k2,	// Value
		&k3,	// Value
		&process,			// DestFinfo
		&sub,				// SharedFinfo
		&prd,				// SharedFinfo
		&enz,				// SharedFinfo
		&cplx,				// SharedFinfo
	};

	static Cinfo enzCinfo (
		"Enz",
		Neutral::initCinfo(),
		enzFinfos,
		sizeof( enzFinfos ) / sizeof ( Finfo* ),
		new Dinfo< Enz >()
	);

	return &enzCinfo;
}

 static const Cinfo* enzCinfo = Enz::initCinfo();

//////////////////////////////////////////////////////////////
// Enz internal functions
//////////////////////////////////////////////////////////////


Enz::Enz( )
	: k1_( 0.1 ), k2_( 0.4 ), k3_( 0.1 )
{
	;
}

//////////////////////////////////////////////////////////////
// MsgDest Definitions
//////////////////////////////////////////////////////////////

void Enz::sub( double n )
{
	sub_ *= n;
}

void Enz::prd( double n ) // dummy
{
	;
}

void Enz::enz( double n ) // dummy
{
	sub_ *= n;
}

void Enz::cplx( double n ) // dummy
{
	cplx_ *= n;
}

void Enz::eprocess( Eref e, const Qinfo* q, ProcInfo* p )
{
	process( p, e );
}

void Enz::process( const ProcInfo* p, const Eref& e )
{
	toSub.send( e, p, cplx_, sub_ );
	toPrd.send( e, p, prd_, 0 );
	toEnz.send( e, p, prd_ + cplx_, sub_ );
	toCplx.send( e, p, sub_, cplx_ );
	
	sub_ = k1_;
	cplx_ = k2_;
	prd_ = k3_;
}

void Enz::reinit( const Eref& e, const Qinfo*q, ProcInfo* p )
{
	;
}

//////////////////////////////////////////////////////////////
// Field Definitions
//////////////////////////////////////////////////////////////

void Enz::setK1( double v )
{
	k1_ = v;
}

double Enz::getK1() const
{
	return k1_;
}

void Enz::setK2( double v )
{
	k2_ = v;
}

double Enz::getK2() const
{
	return k2_;
}

void Enz::setK3( double v )
{
	k3_ = v;
}

double Enz::getK3() const
{
	return k3_;
}

