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
#include "ZombieMMenz.h"
#include "MMenz.h"
#include "DataHandlerWrapper.h"

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
		new OpFunc1< ZombieMMenz, double >( &ZombieMMenz::dummy ) );

static DestFinfo enzDest( "enz",
		"Handles # of molecules of ZombieMMenzyme",
		new OpFunc1< ZombieMMenz, double >( &ZombieMMenz::dummy ) );

static DestFinfo prd( "prdDest",
		"Handles # of molecules of product. Dummy.",
		new OpFunc1< ZombieMMenz, double >( &ZombieMMenz::dummy ) );
	
static Finfo* subShared[] = {
	&toSub, &sub
};

static Finfo* prdShared[] = {
	&toPrd, &prd
};

const Cinfo* ZombieMMenz::initCinfo()
{
		//////////////////////////////////////////////////////////////
		// Field Definitions
		//////////////////////////////////////////////////////////////
		static ElementValueFinfo< ZombieMMenz, double > Km(
			"Km",
			"Michaelis-Menten constant",
			&ZombieMMenz::setKm,
			&ZombieMMenz::getKm
		);

		static ElementValueFinfo< ZombieMMenz, double > kcat(
			"kcat",
			"Forward rate constant for enzyme",
			&ZombieMMenz::setKcat,
			&ZombieMMenz::getKcat
		);

		//////////////////////////////////////////////////////////////
		// MsgDest Definitions
		//////////////////////////////////////////////////////////////
		static DestFinfo process( "process",
			"Handles process call",
			new ProcOpFunc< ZombieMMenz >( &ZombieMMenz::process ) );

		static DestFinfo reinit( "reinit",
			"Handles reinit call",
			new ProcOpFunc< ZombieMMenz >( &ZombieMMenz::reinit ) );

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

	static Cinfo zombieMMenzCinfo (
		"ZombieMMenz",
		Neutral::initCinfo(),
		mmEnzFinfos,
		sizeof( mmEnzFinfos ) / sizeof ( Finfo* ),
		new Dinfo< ZombieMMenz >()
	);

	return &zombieMMenzCinfo;
}

 static const Cinfo* zombieMMenzCinfo = ZombieMMenz::initCinfo();

//////////////////////////////////////////////////////////////
// ZombieMMenz internal functions
//////////////////////////////////////////////////////////////


ZombieMMenz::ZombieMMenz( )
{;}

//////////////////////////////////////////////////////////////
// MsgDest Definitions
//////////////////////////////////////////////////////////////

void ZombieMMenz::dummy( double n )
{;}

void ZombieMMenz::process( const Eref& e, ProcPtr p )
{;}

void ZombieMMenz::reinit( const Eref& e, ProcPtr p )
{;}

//////////////////////////////////////////////////////////////
// Field Definitions
//////////////////////////////////////////////////////////////

void ZombieMMenz::setKm( Eref e, const Qinfo* q, double v )
{
	rates_[ convertIdToMolIndex( e.id() ) ]->setR1( v ); // First rate is Km
}

double ZombieMMenz::getKm( Eref e, const Qinfo* q ) const
{
	return rates_[ convertIdToMolIndex( e.id() ) ]->getR1(); // First rate is Km
}

void ZombieMMenz::setKcat( Eref e, const Qinfo* q, double v )
{
	rates_[ convertIdToMolIndex( e.id() ) ]->setR2( v ); // Second rate is kcat
}

double ZombieMMenz::getKcat( Eref e, const Qinfo* q ) const
{
	return rates_[ convertIdToMolIndex( e.id() ) ]->getR2(); // Second rate is kcat
}

//////////////////////////////////////////////////////////////
// Utility function
//////////////////////////////////////////////////////////////

// static func
void ZombieMMenz::zombify( Element* solver, Element* orig )
{
	static const DestFinfo* enz = dynamic_cast< const DestFinfo* >(
		MMenz::initCinfo()->findFinfo( "enz" ) );
	static const SrcFinfo* sub = dynamic_cast< const SrcFinfo* >(
		MMenz::initCinfo()->findFinfo( "toSub" ) );
	static const SrcFinfo* prd = dynamic_cast< const SrcFinfo* >(
		MMenz::initCinfo()->findFinfo( "toPrd" ) );
	assert( enz );
	assert( sub );
	assert( prd );
	vector< Id > mols;

	Element temp( orig->id(), zombieMMenzCinfo, solver->dataHandler() );
	Eref zer( &temp, 0 );
	Eref oer( orig, 0 );

	ZombieMMenz* z = reinterpret_cast< ZombieMMenz* >( zer.data() );
	MMenz* mmEnz = reinterpret_cast< MMenz* >( oer.data() );

	unsigned int rateIndex = z->convertIdToReacIndex( orig->id() );
	unsigned int num = orig->getInputs( mols, enz );
	const double* enzPtr = &z->S_[ z->convertIdToMolIndex( mols[0] ) ];

	num = orig->getOutputs( mols, sub );
	if ( num == 1 ) {
		const double* subPtr = &z->S_[ z->convertIdToMolIndex( mols[0] ) ];
		assert( num == 1 );
		z->rates_[ rateIndex ] = new MMEnzyme1( 
			mmEnz->getKm(), mmEnz->getKcat(),
			enzPtr, subPtr );
	} else if ( num > 1 ) {
		vector< const double* > v;
		for ( unsigned int i = 0; i < num; ++i )
			v.push_back( &z->S_[ z->convertIdToMolIndex( mols[i] ) ] );
		ZeroOrder* rateTerm = new NOrder( 1.0, v );
		z->rates_[ rateIndex ] = new MMEnzyme( 
			mmEnz->getKm(), mmEnz->getKcat(),
			enzPtr, rateTerm );
	} else {
		cout << "Error: ZombieMMenz::zombify: No substrates\n";
		exit( 0 );
	}

	for ( unsigned int i = 0; i < num; ++i ) {
		unsigned int molIndex = z->convertIdToMolIndex( mols[i] );
		int temp = z->N_.get( molIndex, rateIndex );
		z->N_.set( molIndex, rateIndex, temp - 1 );
	}
	num = orig->getOutputs( mols, prd );
	for ( unsigned int i = 0; i < num; ++i ) {
		unsigned int molIndex = z->convertIdToMolIndex( mols[i] );
		int temp = z->N_.get( molIndex, rateIndex );
		z->N_.set( molIndex, rateIndex, temp + 1 );
	}

	DataHandler* dh = new DataHandlerWrapper( solver->dataHandler() );
	orig->zombieSwap( zombieMMenzCinfo, dh );
}

// Static func
void ZombieMMenz::unzombify( Element* zombie )
{
	Element temp( zombie->id(), zombie->cinfo(), zombie->dataHandler() );
	Eref zer( &temp, 0 );
	Eref oer( zombie, 0 );

	ZombieMMenz* z = reinterpret_cast< ZombieMMenz* >( zer.data() );

	// Here I am unsure how to recreate the correct kind of data handler
	// for the original. Do later.
	DataHandler* dh = 0;

	zombie->zombieSwap( MMenz::initCinfo(), dh );

	MMenz* m = reinterpret_cast< MMenz* >( oer.data() );

	m->setKm( z->getKm( zer, 0 ) );
	m->setKcat( z->getKcat( zer, 0 ) );
}
