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

#define EPSILON 1e-15

static SrcFinfo1< double > nOut( 
		"nOut", 
		"Sends out # of molecules on each timestep"
);

static DestFinfo reacDest( "reacDest",
	"Handles reaction input",
	new OpFunc2< Mol, double, double >( &Mol::reac )
);

static Finfo* reacShared[] = {
	&reacDest, &nOut
};

const Cinfo* Mol::initCinfo()
{
		//////////////////////////////////////////////////////////////
		// Field Definitions
		//////////////////////////////////////////////////////////////
		static ValueFinfo< Mol, double > n(
			"n",
			"Number of molecules",
			&Mol::setN,
			&Mol::getN
		);

		static ValueFinfo< Mol, double > nInit(
			"nInit",
			"Initial value of number of molecules",
			&Mol::setNinit,
			&Mol::getNinit
		);

		static ValueFinfo< Mol, double > conc(
			"conc",
			"Concentration of molecules",
			&Mol::setConc,
			&Mol::getConc
		);

		static ValueFinfo< Mol, double > concInit(
			"concInit",
			"Initial value of molecular concentration",
			&Mol::setConcInit,
			&Mol::getConcInit
		);

		//////////////////////////////////////////////////////////////
		// MsgDest Definitions
		//////////////////////////////////////////////////////////////
		static DestFinfo process( "process",
			"Handles process call",
			new EpFunc1< Mol, ProcPtr >( &Mol::eprocess ) );

		static DestFinfo group( "group",
			"Handle for grouping. Doesn't do anything.",
			new OpFuncDummy() );

		//////////////////////////////////////////////////////////////
		// SharedMsg Definitions
		//////////////////////////////////////////////////////////////
		static SharedFinfo reac( "reac",
			"Connects to reaction",
			reacShared, sizeof( reacShared ) / sizeof( const Finfo* )
		);

	static Finfo* molFinfos[] = {
		&n,	// Value
		&nInit,	// Value
		&process,			// DestFinfo
		&group,			// DestFinfo
		&reac,				// SharedFinfo
	};

	static Cinfo molCinfo (
		"Mol",
		Neutral::initCinfo(),
		molFinfos,
		sizeof( molFinfos ) / sizeof ( Finfo* ),
		new Dinfo< Mol >()
	);

	return &molCinfo;
}

//////////////////////////////////////////////////////////////
// Class definitions
//////////////////////////////////////////////////////////////
static const Cinfo* molCinfo = Mol::initCinfo();

Mol::Mol()
	: n_( 0.0 ), nInit_( 0.0 ), size_( 1.0 ), A_( 0.0 ), B_( 0.0 )
{;}

Mol::Mol( double nInit)
	: n_( 0.0 ), nInit_( nInit ), size_( 1.0 ), A_( 0.0 ), B_( 0.0 )
{;}

//////////////////////////////////////////////////////////////
// MsgDest Definitions
//////////////////////////////////////////////////////////////

void Mol::eprocess( Eref e, const Qinfo* q, ProcInfo* p )
{
	process( p, e );
}

void Mol::process( const ProcInfo* p, const Eref& e )
{
	// double A = e.sumBuf( aSlot );
	// double B = e.sumBuf( bSlot );
	if ( n_ > EPSILON && B_ > EPSILON ) {
		double C = exp( -B_ * p->dt / n_ );
		n_ *= C + (A_ / B_ ) * ( 1.0 - C );
	} else {
		n_ += ( A_ - B_ ) * p->dt;
	}

	A_ = B_ = 0.0;

	nOut.send( e, p, n_ );
}

void Mol::reac( double A, double B )
{
	A_ += A;
	B_ += B;
}

void Mol::reinit( const Eref& e, const Qinfo*q, ProcInfo* p )
{
	n_ = nInit_;

	nOut.send( e, p, n_ );
}

//////////////////////////////////////////////////////////////
// Field Definitions
//////////////////////////////////////////////////////////////

void Mol::setN( double v )
{
	n_ = v;
}

double Mol::getN() const
{
	return n_;
}

void Mol::setNinit( double v )
{
	nInit_ = v;
}

double Mol::getNinit() const
{
	return nInit_;
}

void Mol::setConc( double v )
{
	n_ = v * size_;
}

double Mol::getConc() const
{
	return n_ / size_;
}

void Mol::setConcInit( double v )
{
	nInit_ = v * size_;
}

double Mol::getConcInit() const
{
	return nInit_ / size_;
}
