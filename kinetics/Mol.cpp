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

		//////////////////////////////////////////////////////////////
		// MsgDest Definitions
		//////////////////////////////////////////////////////////////
		static DestFinfo process( "process",
			"Handles process call",
			new EpFunc1< Mol, ProcPtr >( &Mol::eprocess ) );

		static DestFinfo reac( "reac",
			"Handles reaction input",
			new OpFunc2< Mol, double, double >( &Mol::reac ) );

		static DestFinfo sumTotal( "sumTotal",
			"Handles summing input. Deprecated",
			new OpFunc1< Mol, double >( &Mol::sumTotal ) );

	static Finfo* molFinfos[] = {
		&n,	// Value
		&nInit,	// Value
		&process,			// DestFinfo
		&reac,				// DestFinfo
		&sumTotal,			// DestFinfo
		&nOut,				// SrcFinfo
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

void Mol::sumTotal( double v )
{
	;
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
