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
#include "ZombieReac.h"
#include "ElementValueFinfo.h"

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
		new OpFunc1< ZombieReac, double >( &ZombieReac::sub ) );

static DestFinfo prd( "prdDest",
		"Handles # of molecules of product",
		new OpFunc1< ZombieReac, double >( &ZombieReac::prd ) );
	
static Finfo* subShared[] = {
	&toSub, &sub
};

static Finfo* prdShared[] = {
	&toPrd, &prd
};

const Cinfo* ZombieReac::initCinfo()
{
		//////////////////////////////////////////////////////////////
		// Field Definitions
		//////////////////////////////////////////////////////////////
		static ElementValueFinfo< ZombieReac, double > kf(
			"kf",
			"Forward rate constant",
			&ZombieReac::setKf,
			&ZombieReac::getKf
		);

		static ElementValueFinfo< ZombieReac, double > kb(
			"kb",
			"Backward rate constant",
			&ZombieReac::setKb,
			&ZombieReac::getKb
		);

		//////////////////////////////////////////////////////////////
		// MsgDest Definitions
		//////////////////////////////////////////////////////////////
		static DestFinfo process( "process",
			"Handles process call",
			new EpFunc1< ZombieReac, ProcPtr >( &ZombieReac::eprocess ) );

		static DestFinfo group( "group",
			"Handle for grouping. Doesn't do anything.",
			new OpFuncDummy() );

		//////////////////////////////////////////////////////////////
		// SharedMsg Definitions
		//////////////////////////////////////////////////////////////
		static SharedFinfo sub( "sub",
			"Connects to substrate molecule",
			subShared, sizeof( subShared ) / sizeof( const Finfo* )
		);
		static SharedFinfo prd( "prd",
			"Connects to substrate molecule",
			prdShared, sizeof( prdShared ) / sizeof( const Finfo* )
		);

	static Finfo* zombieReacFinfos[] = {
		&kf,		// Value
		&kb,		// Value
		&process,	// DestFinfo
		&sub,		// SharedFinfo
		&prd,		// SharedFinfo
	};

	static Cinfo zombieReacCinfo (
		"ZombieReac",
		Neutral::initCinfo(),
		zombieReacFinfos,
		sizeof( zombieReacFinfos ) / sizeof ( Finfo* ),
		new Dinfo< ZombieReac >()
	);

	return &zombieReacCinfo;
}

//////////////////////////////////////////////////////////////
// Class definitions
//////////////////////////////////////////////////////////////
static const Cinfo* zombieReacCinfo = ZombieReac::initCinfo();

ZombieReac::ZombieReac()
{;}

ZombieReac::~ZombieReac()
{;}


//////////////////////////////////////////////////////////////
// MsgDest Definitions
//////////////////////////////////////////////////////////////

// Doesn't do anything on its own.
void ZombieReac::eprocess( Eref e, const Qinfo* q, ProcInfo* p )
{
}

void ZombieReac::process( const ProcInfo* p, const Eref& e )
{
}

void ZombieReac::sub( double v )
{
}

void ZombieReac::prd( double v )
{
}

//////////////////////////////////////////////////////////////
// Field Definitions
//////////////////////////////////////////////////////////////

unsigned int  ZombieReac::convertId( Id id ) const
{
	unsigned int i = id.value() - objMapStart_;
	assert( i < objMap_.size() );
	i = objMap_[i];
	assert( i < rates_.size() );
	return i;
}

void ZombieReac::setKf( Eref e, const Qinfo* q, double v )
{
	rates_[ convertId( e.id() ) ]->setR1( v );
}

double ZombieReac::getKf( Eref e, const Qinfo* q ) const
{
	return rates_[ convertId( e.id() ) ]->getR1();
}

void ZombieReac::setKb( Eref e, const Qinfo* q, double v )
{
	rates_[ convertId( e.id() ) ]->setR2( v );
}

double ZombieReac::getKb( Eref e, const Qinfo* q ) const
{
	return rates_[ convertId( e.id() ) ]->getR2();
}
