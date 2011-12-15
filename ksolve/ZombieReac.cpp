/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "StoichHeaders.h"
#include "ZombieReac.h"
#include "Reac.h"
#include "ElementValueFinfo.h"
#include "DataHandlerWrapper.h"

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

static SrcFinfo1< double > *requestSize() {
	static SrcFinfo1< double > requestSize( 
			"requestSize", 
			"Requests size (volume) in which reaction is embedded. Used for"
			"conversion to concentration units from molecule # units,"
			"and for calculations when resized."
			);
	return &requestSize;
}

const Cinfo* ZombieReac::initCinfo()
{
		//////////////////////////////////////////////////////////////
		// Field Definitions
		//////////////////////////////////////////////////////////////
		static ElementValueFinfo< ZombieReac, double > kf(
			"kf",
			"Forward rate constant, in # units",
			&ZombieReac::setKf,
			&ZombieReac::getKf
		);

		static ElementValueFinfo< ZombieReac, double > kb(
			"kb",
			"Reverse rate constant, in # units",
			&ZombieReac::setKb,
			&ZombieReac::getKb
		);

		static ElementValueFinfo< ZombieReac, double > Kf(
			"Kf",
			"Forward rate constant, in concentration units",
			&ZombieReac::setConcKf,
			&ZombieReac::getConcKf
		);

		static ElementValueFinfo< ZombieReac, double > Kb(
			"Kb",
			"Reverse rate constant, in concentration units",
			&ZombieReac::setConcKb,
			&ZombieReac::getConcKb
		);

		//////////////////////////////////////////////////////////////
		// MsgDest Definitions
		//////////////////////////////////////////////////////////////
		static DestFinfo process( "process",
			"Handles process call",
			new ProcOpFunc< ZombieReac >( &ZombieReac::process ) );

		static DestFinfo reinit( "reinit",
			"Handles reinit call",
			new ProcOpFunc< ZombieReac >( &ZombieReac::reinit ) );

		static DestFinfo group( "group",
			"Handle for grouping. Doesn't do anything.",
			new OpFuncDummy() );

		//////////////////////////////////////////////////////////////
		// SharedMsg Definitions
		//////////////////////////////////////////////////////////////
		static DestFinfo subDest( "subDest",
				"Handles # of molecules of substrate",
				new OpFunc1< ZombieReac, double >( &ZombieReac::sub ) );
		static DestFinfo prdDest( "prdDest",
				"Handles # of molecules of product",
				new OpFunc1< ZombieReac, double >( &ZombieReac::prd ) );
		static Finfo* subShared[] = {
			toSub(), &subDest
		};
		static Finfo* prdShared[] = {
			toPrd(), &prdDest
		};
		static SharedFinfo sub( "sub",
			"Connects to substrate molecule",
			subShared, sizeof( subShared ) / sizeof( const Finfo* )
		);
		static SharedFinfo prd( "prd",
			"Connects to substrate molecule",
			prdShared, sizeof( prdShared ) / sizeof( const Finfo* )
		);
		static Finfo* procShared[] = {
			&process, &reinit
		};
		static SharedFinfo proc( "proc",
			"Shared message for process and reinit",
			procShared, sizeof( procShared ) / sizeof( const Finfo* )
		);

	static Finfo* zombieReacFinfos[] = {
		&kf,		// Value
		&kb,		// Value
		&Kf,		// Value
		&Kb,		// Value
		requestSize(),	// SrcFinfo
		&sub,		// SharedFinfo
		&prd,		// SharedFinfo
		&proc,		// SharedFinfo
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
void ZombieReac::process( const Eref& e, ProcPtr p )
{;}

void ZombieReac::reinit( const Eref& e, ProcPtr p )
{;}


void ZombieReac::sub( double v )
{
}

void ZombieReac::prd( double v )
{
}

//////////////////////////////////////////////////////////////
// Field Definitions
//////////////////////////////////////////////////////////////

void ZombieReac::setKf( const Eref& e, const Qinfo* q, double v )
{
	rates_[ convertIdToReacIndex( e.id() ) ]->setR1( v );
}

double ZombieReac::getKf( const Eref& e, const Qinfo* q ) const
{
	return rates_[ convertIdToReacIndex( e.id() ) ]->getR1();
}

void ZombieReac::setKb( const Eref& e, const Qinfo* q, double v )
{
	if ( useOneWay_ )
		rates_[ convertIdToReacIndex( e.id() ) + 1 ]->setR1( v );
	else
		rates_[ convertIdToReacIndex( e.id() ) ]->setR2( v );
}

double ZombieReac::getKb( const Eref& e, const Qinfo* q ) const
{
	if ( useOneWay_ )
		return rates_[ convertIdToReacIndex( e.id() ) + 1 ]->getR1();
	else
		return rates_[ convertIdToReacIndex( e.id() ) ]->getR2();
}

void ZombieReac::setConcKf( const Eref& e, const Qinfo* q, double v )
{
	double volScale = 
		convertConcToNumRateUsingMesh( e, toSub(), 0, 1.0e-3, 0 );

	setKf( e, q, v * volScale );
}

double ZombieReac::getConcKf( const Eref& e, const Qinfo* q ) const
{
	double volScale = 
		convertConcToNumRateUsingMesh( e, toSub(), 0, 1.0e-3, 0 );
	return getKf( e, q ) / volScale;
}

void ZombieReac::setConcKb( const Eref& e, const Qinfo* q, double v )
{
	double volScale = 
		convertConcToNumRateUsingMesh( e, toPrd(), 0, 1.0e-3, 0 );
	setKb( e, q, v * volScale );
}

double ZombieReac::getConcKb( const Eref& e, const Qinfo* q ) const
{
	double volScale = 
		convertConcToNumRateUsingMesh( e, toPrd(), 0, 1.0e-3, 0 );
	return getKb( e, q ) / volScale;
}

//////////////////////////////////////////////////////////////
// Utility function
//////////////////////////////////////////////////////////////

ZeroOrder* ZombieReac::makeHalfReaction( 
	Element* orig, double rate, const SrcFinfo* finfo ) const
{
	vector< Id > mols;
	unsigned int numReactants = orig->getOutputs( mols, finfo ); 
	ZeroOrder* rateTerm = 0;
	if ( numReactants == 1 ) {
		rateTerm = 
			new FirstOrder( rate, convertIdToPoolIndex( mols[0] ) );
	} else if ( numReactants == 2 ) {
		rateTerm = new SecondOrder( rate,
				convertIdToPoolIndex( mols[0] ), 
				convertIdToPoolIndex( mols[1] ) );
	} else if ( numReactants > 2 ) {
		vector< unsigned int > v;
		for ( unsigned int i = 0; i < numReactants; ++i ) {
			v.push_back( convertIdToPoolIndex( mols[i] ) );
		}
		rateTerm = new NOrder( rate, v );
	} else {
		cout << "Error: ZombieReac::makeHalfReaction: zero reactants\n";
	}
	return rateTerm;
}

// static func
void ZombieReac::zombify( Element* solver, Element* orig )
{
	static const SrcFinfo* sub = dynamic_cast< const SrcFinfo* >(
		Reac::initCinfo()->findFinfo( "toSub" ) );
	static const SrcFinfo* prd = dynamic_cast< const SrcFinfo* >(
		Reac::initCinfo()->findFinfo( "toPrd" ) );
	
	assert( sub );
	assert( prd );

	Element temp( orig->id(), zombieReacCinfo, solver->dataHandler() );
	Eref zer( &temp, 0 );
	Eref oer( orig, 0 );

	ZombieReac* z = reinterpret_cast< ZombieReac* >( zer.data() );
	Reac* reac = reinterpret_cast< Reac* >( oer.data() );

	ZeroOrder* forward = z->makeHalfReaction( orig, reac->getKf(), sub );
	ZeroOrder* reverse = z->makeHalfReaction( orig, reac->getKb(), prd );

	unsigned int rateIndex = z->convertIdToReacIndex( orig->id() );
	unsigned int revRateIndex = rateIndex;
	if ( z->useOneWay_ ) {
		z->rates_[ rateIndex ] = forward;
		revRateIndex = rateIndex + 1;
		z->rates_[ revRateIndex ] = reverse;
	} else {
		z->rates_[ rateIndex ] = 
			new BidirectionalReaction( forward, reverse );
	}

	vector< unsigned int > molIndex;

	if ( z->useOneWay_ ) {
		unsigned int numReactants = forward->getReactants( molIndex );
		for ( unsigned int i = 0; i < numReactants; ++i ) {
			int temp = z->N_.get( molIndex[i], rateIndex );
			z->N_.set( molIndex[i], rateIndex, temp - 1 );
			temp = z->N_.get( molIndex[i], revRateIndex );
			z->N_.set( molIndex[i], revRateIndex, temp + 1 );
		}

		numReactants = reverse->getReactants( molIndex );
		for ( unsigned int i = 0; i < numReactants; ++i ) {
			int temp = z->N_.get( molIndex[i], rateIndex );
			z->N_.set( molIndex[i], rateIndex, temp + 1 );
			temp = z->N_.get( molIndex[i], revRateIndex );
			z->N_.set( molIndex[i], revRateIndex, temp - 1 );
		}
	} else {
		unsigned int numReactants = forward->getReactants( molIndex );
		for ( unsigned int i = 0; i < numReactants; ++i ) {
			int temp = z->N_.get( molIndex[i], rateIndex );
			z->N_.set( molIndex[i], rateIndex, temp - 1 );
		}

		numReactants = reverse->getReactants( molIndex );
		for ( unsigned int i = 0; i < numReactants; ++i ) {
			int temp = z->N_.get( molIndex[i], revRateIndex );
			z->N_.set( molIndex[i], rateIndex, temp + 1 );
		}
	}

	DataHandler* dh = new DataHandlerWrapper( solver->dataHandler(),
		orig->dataHandler() );
	orig->zombieSwap( zombieReacCinfo, dh );
}

// Static func
void ZombieReac::unzombify( Element* zombie )
{
	Element temp( zombie->id(), zombie->cinfo(), zombie->dataHandler() );
	Eref zer( &temp, 0 );
	Eref oer( zombie, 0 );

	ZombieReac* z = reinterpret_cast< ZombieReac* >( zer.data() );

	// Here I am unsure how to recreate the correct kind of data handler
	// for the original. Do later.
	DataHandler* dh = 0;

	zombie->zombieSwap( Reac::initCinfo(), dh );

	Reac* m = reinterpret_cast< Reac* >( oer.data() );

	m->setKf( z->getKf( zer, 0 ) );
	m->setKb( z->getKb( zer, 0 ) );
}
