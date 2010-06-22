/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
#include "RateTerm.h"
#include "SparseMatrix.h"
#include "KinSparseMatrix.h"
#include "Stoich.h"
#include "ZombieMol.h"
#include "ElementValueFinfo.h"

#define EPSILON 1e-15

static SrcFinfo1< double > nOut( 
		"nOut", 
		"Sends out # of molecules on each timestep"
);

static DestFinfo reacDest( "reacDest",
	"Handles reaction input",
	new OpFunc2< ZombieMol, double, double >( &ZombieMol::reac )
);

static Finfo* reacShared[] = {
	&reacDest, &nOut
};

const Cinfo* ZombieMol::initCinfo()
{
		//////////////////////////////////////////////////////////////
		// Field Definitions
		//////////////////////////////////////////////////////////////
		static ElementValueFinfo< ZombieMol, double > n(
			"n",
			"Number of molecules",
			&ZombieMol::setN,
			&ZombieMol::getN
		);

		static ElementValueFinfo< ZombieMol, double > nInit(
			"nInit",
			"Initial value of number of molecules",
			&ZombieMol::setNinit,
			&ZombieMol::getNinit
		);

		static ElementValueFinfo< ZombieMol, double > diffConst(
			"diffConst",
			"Diffusion constant of molecule",
			&ZombieMol::setDiffConst,
			&ZombieMol::getDiffConst
		);

		static ElementValueFinfo< ZombieMol, double > conc(
			"conc",
			"Concentration of molecules",
			&ZombieMol::setConc,
			&ZombieMol::getConc
		);

		static ElementValueFinfo< ZombieMol, double > concInit(
			"concInit",
			"Initial value of molecular concentration",
			&ZombieMol::setConcInit,
			&ZombieMol::getConcInit
		);

		//////////////////////////////////////////////////////////////
		// MsgDest Definitions
		//////////////////////////////////////////////////////////////
		static DestFinfo process( "process",
			"Handles process call",
			new EpFunc1< ZombieMol, ProcPtr >( &ZombieMol::eprocess ) );

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

	static Finfo* zombieMolFinfos[] = {
		&n,	// Value
		&nInit,	// Value
		&diffConst,	// Value
		&process,			// DestFinfo
		&group,			// DestFinfo
		&reac,				// SharedFinfo
	};

	static Cinfo zombieMolCinfo (
		"ZombieMol",
		Neutral::initCinfo(),
		zombieMolFinfos,
		sizeof( zombieMolFinfos ) / sizeof ( Finfo* ),
		new Dinfo< ZombieMol >()
	);

	return &zombieMolCinfo;
}

//////////////////////////////////////////////////////////////
// Class definitions
//////////////////////////////////////////////////////////////
static const Cinfo* zombieMolCinfo = ZombieMol::initCinfo();

ZombieMol::ZombieMol()
{;}

ZombieMol::~ZombieMol()
{;}


//////////////////////////////////////////////////////////////
// MsgDest Definitions
//////////////////////////////////////////////////////////////

// Doesn't do anything on its own.
void ZombieMol::eprocess( Eref e, const Qinfo* q, ProcInfo* p )
{
}

void ZombieMol::process( const ProcInfo* p, const Eref& e )
{
}

void ZombieMol::reac( double A, double B )
{
}

void ZombieMol::reinit( const Eref& e, const Qinfo*q, ProcInfo* p )
{
}

//////////////////////////////////////////////////////////////
// Field Definitions
//////////////////////////////////////////////////////////////

unsigned int  ZombieMol::convertId( Id id ) const
{
	unsigned int i = id.value() - objMapStart_;
	assert( i < objMap_.size() );
	i = objMap_[i];
	assert( i < S_.size() );
	return i;
}


void ZombieMol::setN( Eref e, const Qinfo* q, double v )
{
	S_[ convertId( e.id() ) ] = v;
}

double ZombieMol::getN( Eref e, const Qinfo* q ) const
{
	return S_[ convertId( e.id() ) ];
}

void ZombieMol::setNinit( Eref e, const Qinfo* q, double v )
{
	Sinit_[ convertId( e.id() ) ] = v;
}

double ZombieMol::getNinit( Eref e, const Qinfo* q ) const
{
	return Sinit_[ convertId( e.id() ) ];
}

void ZombieMol::setConc( Eref e, const Qinfo* q, double v )
{
	// n_ = v * size_;
}

double ZombieMol::getConc( Eref e, const Qinfo* q ) const
{
	// return n_ / size_;
	return 0;
}

void ZombieMol::setConcInit( Eref e, const Qinfo* q, double v )
{
	// nInit_ = v * size_;
}

double ZombieMol::getConcInit( Eref e, const Qinfo* q ) const
{
	// return nInit_ / size_;
	return 0;
}

void ZombieMol::setDiffConst( Eref e, const Qinfo* q, double v )
{
	// diffConst_ = v;
}

double ZombieMol::getDiffConst( Eref e, const Qinfo* q ) const
{
	// return diffConst_;
	return 0;
}
