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
#include "ElementValueFinfo.h"
#include "ZombieEnz.h"
#include "Enz.h"
#include "DataHandlerWrapper.h"

static SrcFinfo2< double, double > toSub( 
		"toSub", 
		"Sends out increment of molecules on product each timestep"
	);

static SrcFinfo2< double, double > toPrd( 
		"toPrd", 
		"Sends out increment of molecules on product each timestep"
	);
	
static SrcFinfo2< double, double > toZombieEnz( 
		"toZombieEnz", 
		"Sends out increment of molecules on product each timestep"
	);
static SrcFinfo2< double, double > toCplx( 
		"toCplx", 
		"Sends out increment of molecules on product each timestep"
	);

static DestFinfo sub( "subDest",
		"Handles # of molecules of substrate",
		new OpFunc1< ZombieEnz, double >( &ZombieEnz::dummy ) );

static DestFinfo enz( "enzDest",
		"Handles # of molecules of ZombieEnzyme",
		new OpFunc1< ZombieEnz, double >( &ZombieEnz::dummy ) );

static DestFinfo prd( "prdDest",
		"Handles # of molecules of product. Dummy.",
		new OpFunc1< ZombieEnz, double >( &ZombieEnz::dummy ) );

static DestFinfo cplx( "prdDest",
		"Handles # of molecules of enz-sub complex",
		new OpFunc1< ZombieEnz, double >( &ZombieEnz::dummy ) );
	
static Finfo* subShared[] = {
	&toSub, &sub
};

static Finfo* enzShared[] = {
	&toZombieEnz, &enz
};

static Finfo* prdShared[] = {
	&toPrd, &prd
};

static Finfo* cplxShared[] = {
	&toCplx, &cplx
};

const Cinfo* ZombieEnz::initCinfo()
{
		//////////////////////////////////////////////////////////////
		// Field Definitions
		//////////////////////////////////////////////////////////////
		static ElementValueFinfo< ZombieEnz, double > k1(
			"k1",
			"Forward rate constant",
			&ZombieEnz::setK1,
			&ZombieEnz::getK1
		);

		static ElementValueFinfo< ZombieEnz, double > k2(
			"k2",
			"Forward rate constant",
			&ZombieEnz::setK2,
			&ZombieEnz::getK2
		);

		static ElementValueFinfo< ZombieEnz, double > k3(
			"k3",
			"Forward rate constant",
			&ZombieEnz::setK3,
			&ZombieEnz::getK3
		);

		//////////////////////////////////////////////////////////////
		// MsgDest Definitions
		//////////////////////////////////////////////////////////////
		static DestFinfo process( "process",
			"Handles process call",
			new EpFunc1< ZombieEnz, ProcPtr >( &ZombieEnz::eprocess ) );

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

	static Finfo* zombieEnzFinfos[] = {
		&k1,	// Value
		&k2,	// Value
		&k3,	// Value
		&process,			// DestFinfo
		&sub,				// SharedFinfo
		&prd,				// SharedFinfo
		&enz,				// SharedFinfo
		&cplx,				// SharedFinfo
	};

	static Cinfo zombieEnzCinfo (
		"ZombieEnz",
		Neutral::initCinfo(),
		zombieEnzFinfos,
		sizeof( zombieEnzFinfos ) / sizeof ( Finfo* ),
		new Dinfo< ZombieEnz >()
	);

	return &zombieEnzCinfo;
}

 static const Cinfo* zombieEnzCinfo = ZombieEnz::initCinfo();

//////////////////////////////////////////////////////////////
// ZombieEnz internal functions
//////////////////////////////////////////////////////////////

ZombieEnz::ZombieEnz( )
{ ; }

//////////////////////////////////////////////////////////////
// MsgDest Definitions
//////////////////////////////////////////////////////////////

void ZombieEnz::dummy( double n ) // dummy
{;}

void ZombieEnz::eprocess( Eref e, const Qinfo* q, ProcInfo* p )
{;}

void ZombieEnz::process( const ProcInfo* p, const Eref& e )
{;}

void ZombieEnz::reinit( const Eref& e, const Qinfo*q, ProcInfo* p )
{;}

//////////////////////////////////////////////////////////////
// Field Definitions
//////////////////////////////////////////////////////////////

void ZombieEnz::setK1( Eref e, const Qinfo* q, double v )
{
	rates_[ convertId( e.id() ) ]->setR1( v );
}

double ZombieEnz::getK1( Eref e, const Qinfo* q ) const
{
	return rates_[ convertId( e.id() ) ]->getR1();
}

void ZombieEnz::setK2( Eref e, const Qinfo* q, double v )
{
	rates_[ convertId( e.id() ) ]->setR2( v );
}

double ZombieEnz::getK2( Eref e, const Qinfo* q ) const
{
	return rates_[ convertId( e.id() ) ]->getR2();
}

void ZombieEnz::setK3( Eref e, const Qinfo* q, double v )
{
	rates_[ convertId( e.id() ) + 1 ]->setR1( v );
}

double ZombieEnz::getK3( Eref e, const Qinfo* q ) const
{
	return rates_[ convertId( e.id() ) + 1 ]->getR1();
}

//////////////////////////////////////////////////////////////
// Utility function
//////////////////////////////////////////////////////////////

unsigned int  ZombieEnz::convertId( Id id ) const
{
	unsigned int i = id.value() - objMapStart_;
	assert( i < objMap_.size() );
	i = objMap_[i];
	assert( ( i + 1 ) < rates_.size() ); // enz uses two rate terms.
	return i;
}

// static func
void ZombieEnz::zombify( Element* solver, Element* orig )
{
	Element temp( orig->id(), zombieEnzCinfo, solver->dataHandler() );
	Eref zer( &temp, 0 );
	Eref oer( orig, 0 );

	ZombieEnz* z = reinterpret_cast< ZombieEnz* >( zer.data() );
	Enz* enz = reinterpret_cast< Enz* >( oer.data() );

	z->setK1( zer, 0, enz->getK1() );
	z->setK2( zer, 0, enz->getK2() );
	z->setK3( zer, 0, enz->getK3() );
	DataHandler* dh = new DataHandlerWrapper( solver->dataHandler() );
	orig->zombieSwap( zombieEnzCinfo, dh );
}

// Static func
void ZombieEnz::unzombify( Element* zombie )
{
	Element temp( zombie->id(), zombie->cinfo(), zombie->dataHandler() );
	Eref zer( &temp, 0 );
	Eref oer( zombie, 0 );

	ZombieEnz* z = reinterpret_cast< ZombieEnz* >( zer.data() );

	// Here I am unsure how to recreate the correct kind of data handler
	// for the original. Do later.
	DataHandler* dh = 0;

	zombie->zombieSwap( Enz::initCinfo(), dh );

	Enz* m = reinterpret_cast< Enz* >( oer.data() );

	m->setK1( z->getK1( zer, 0 ) );
	m->setK2( z->getK2( zer, 0 ) );
	m->setK3( z->getK3( zer, 0 ) );
}
