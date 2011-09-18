/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "StoichHeaders.h"
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
			"Connects to substrate pool",
			subShared, sizeof( subShared ) / sizeof( const Finfo* )
		);
		static SharedFinfo prd( "prd",
			"Connects to product pool",
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

double getEnzVol( const Eref& e )
{
	vector< Id > enzMol;
	e.element()->getInputs( enzMol, &enzDest );
	assert( enzMol.size() == 1 );
	const Finfo* f1 = enzMol[0].element()->cinfo()->findFinfo( "requestSize" );
	const SrcFinfo* sf = dynamic_cast< const SrcFinfo* >( f1 );
	assert( sf );
	double vol = lookupSizeFromMesh( enzMol[0].eref(), sf );
	assert( vol > 0.0 );
	return vol;
}

void ZombieMMenz::setKm( const Eref& e, const Qinfo* q, double v )
{
	double Km = v * NA * CONC_UNIT_CONV * getEnzVol( e );

	// First rate is Km
	rates_[ convertIdToPoolIndex( e.id() ) ]->setR1( Km ); 
}

double ZombieMMenz::getKm( const Eref& e, const Qinfo* q ) const
{
	double Km = 
		rates_[ convertIdToPoolIndex( e.id() ) ]->getR1();
	
	return Km / getEnzVol( e ) * NA * CONC_UNIT_CONV;
}

void ZombieMMenz::setKcat( const Eref& e, const Qinfo* q, double v )
{
	rates_[ convertIdToPoolIndex( e.id() ) ]->setR2( v ); // Second rate is kcat
}

double ZombieMMenz::getKcat( const Eref& e, const Qinfo* q ) const
{
	return rates_[ convertIdToPoolIndex( e.id() ) ]->getR2(); // Second rate is kcat
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
	vector< Id > pools;

	Element temp( orig->id(), zombieMMenzCinfo, solver->dataHandler() );
	Eref zer( &temp, 0 );
	Eref oer( orig, 0 );

	ZombieMMenz* z = reinterpret_cast< ZombieMMenz* >( zer.data() );
	MMenz* mmEnz = reinterpret_cast< MMenz* >( oer.data() );

	unsigned int rateIndex = z->convertIdToReacIndex( orig->id() );
	unsigned int num = orig->getInputs( pools, enz );
	unsigned int enzIndex = z->convertIdToPoolIndex( pools[0] );

	num = orig->getOutputs( pools, sub );
	if ( num == 1 ) {
		unsigned int subIndex = z->convertIdToPoolIndex( pools[0] );
		assert( num == 1 );
		z->rates_[ rateIndex ] = new MMEnzyme1( 
			mmEnz->getKm(), mmEnz->getKcat(),
			enzIndex, subIndex );
	} else if ( num > 1 ) {
		vector< unsigned int > v;
		for ( unsigned int i = 0; i < num; ++i )
			v.push_back( z->convertIdToPoolIndex( pools[i] ) );
		ZeroOrder* rateTerm = new NOrder( 1.0, v );
		z->rates_[ rateIndex ] = new MMEnzyme( 
			mmEnz->getKm(), mmEnz->getKcat(),
			enzIndex, rateTerm );
	} else {
		cout << "Error: ZombieMMenz::zombify: No substrates\n";
		exit( 0 );
	}

	for ( unsigned int i = 0; i < num; ++i ) {
		unsigned int poolIndex = z->convertIdToPoolIndex( pools[i] );
		int temp = z->N_.get( poolIndex, rateIndex );
		z->N_.set( poolIndex, rateIndex, temp - 1 );
	}
	num = orig->getOutputs( pools, prd );
	for ( unsigned int i = 0; i < num; ++i ) {
		unsigned int poolIndex = z->convertIdToPoolIndex( pools[i] );
		int temp = z->N_.get( poolIndex, rateIndex );
		z->N_.set( poolIndex, rateIndex, temp + 1 );
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
