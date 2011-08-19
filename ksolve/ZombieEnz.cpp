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
			new ProcOpFunc< ZombieEnz >( &ZombieEnz::process ) );

		static DestFinfo reinit( "reinit",
			"Handles reinit call",
			new ProcOpFunc< ZombieEnz >( &ZombieEnz::reinit ) );

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
		static SharedFinfo enz( "enz",
			"Connects to enzyme pool",
			enzShared, sizeof( enzShared ) / sizeof( const Finfo* )
		);
		static SharedFinfo cplx( "cplx",
			"Connects to enz-sub complex pool",
			cplxShared, sizeof( cplxShared ) / sizeof( const Finfo* )
		);
		static Finfo* procShared[] = {
			&process, &reinit
		};
		static SharedFinfo proc( "proc",
			"Shared message for process and reinit",
			procShared, sizeof( procShared ) / sizeof( const Finfo* )
		);

	static Finfo* zombieEnzFinfos[] = {
		&k1,	// Value
		&k2,	// Value
		&k3,	// Value
		&sub,				// SharedFinfo
		&prd,				// SharedFinfo
		&enz,				// SharedFinfo
		&cplx,				// SharedFinfo
		&proc,				// SharedFinfo
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

void ZombieEnz::process( const Eref& e, ProcPtr p )
{;}

void ZombieEnz::reinit( const Eref& e, ProcPtr p )
{;}

//////////////////////////////////////////////////////////////
// Field Definitions
//////////////////////////////////////////////////////////////

void ZombieEnz::setK1( const Eref& e, const Qinfo* q, double v )
{
	rates_[ convertIdToReacIndex( e.id() ) ]->setR1( v );
}

double ZombieEnz::getK1( const Eref& e, const Qinfo* q ) const
{
	return rates_[ convertIdToReacIndex( e.id() ) ]->getR1();
}

void ZombieEnz::setK2( const Eref& e, const Qinfo* q, double v )
{
	if ( useOneWay_ )
		rates_[ convertIdToReacIndex( e.id() ) + 1 ]->setR1( v );
	else
		rates_[ convertIdToReacIndex( e.id() ) ]->setR2( v );
}

double ZombieEnz::getK2( const Eref& e, const Qinfo* q ) const
{
	if ( useOneWay_ )
		return rates_[ convertIdToReacIndex( e.id() ) + 1 ]->getR1();
	else
		return rates_[ convertIdToReacIndex( e.id() ) ]->getR2();
}

void ZombieEnz::setK3( const Eref& e, const Qinfo* q, double v )
{
	if ( useOneWay_ )
		rates_[ convertIdToReacIndex( e.id() ) + 2 ]->setR1( v );
	else
		rates_[ convertIdToReacIndex( e.id() ) + 1 ]->setR1( v );
}

double ZombieEnz::getK3( const Eref& e, const Qinfo* q ) const
{
	if ( useOneWay_ )
		return rates_[ convertIdToReacIndex( e.id() ) + 2 ]->getR1();
	else
		return rates_[ convertIdToReacIndex( e.id() ) + 1 ]->getR1();
}

//////////////////////////////////////////////////////////////
// Utility function
//////////////////////////////////////////////////////////////

ZeroOrder* ZombieEnz::makeHalfReaction( 
	Element* orig, double rate, const SrcFinfo* finfo, Id enz ) const
{
	vector< Id > pools;
	unsigned int numReactants = orig->getOutputs( pools, finfo ); 
	if ( enz != Id() ) // Used to add the enz to the reactants.
		pools.push_back( enz );
	numReactants = pools.size();

	ZeroOrder* rateTerm = 0;
	if ( numReactants == 1 ) {
		rateTerm = 
			new FirstOrder( rate, convertIdToPoolIndex( pools[0] ) );
	} else if ( numReactants == 2 ) {
		rateTerm = new SecondOrder( rate,
				convertIdToPoolIndex( pools[0] ), 
				convertIdToPoolIndex( pools[1] ) );
	} else if ( numReactants > 2 ) {
		vector< unsigned int > v;
		for ( unsigned int i = 0; i < numReactants; ++i ) {
			v.push_back( convertIdToPoolIndex( pools[i] ) );
		}
		rateTerm = new NOrder( rate, v );
	} else {
		cout << "Error: ZombieEnz::makeHalfReaction: zero reactants\n";
	}
	return rateTerm;
}

// static func
void ZombieEnz::zombify( Element* solver, Element* orig )
{
	static const SrcFinfo* sub = dynamic_cast< const SrcFinfo* >(
		Enz::initCinfo()->findFinfo( "toSub" ) );
	static const SrcFinfo* prd = dynamic_cast< const SrcFinfo* >(
		Enz::initCinfo()->findFinfo( "toPrd" ) );
	static const SrcFinfo* enzFinfo = dynamic_cast< const SrcFinfo* >(
		Enz::initCinfo()->findFinfo( "toEnz" ) );
	static const SrcFinfo* cplx = dynamic_cast< const SrcFinfo* >(
		Enz::initCinfo()->findFinfo( "toCplx" ) );

	Element temp( orig->id(), zombieEnzCinfo, solver->dataHandler() );
	Eref zer( &temp, 0 );
	Eref oer( orig, 0 );

	ZombieEnz* z = reinterpret_cast< ZombieEnz* >( zer.data() );
	Enz* enz = reinterpret_cast< Enz* >( oer.data() );

	vector< Id > pools;
	unsigned int numReactants = orig->getOutputs( pools, enzFinfo ); 
	assert( numReactants == 1 );
	Id enzId = pools[0];

	ZeroOrder* r1 = z->makeHalfReaction( orig, enz->getK1(), sub, enzId );
	ZeroOrder* r2 = z->makeHalfReaction( orig, enz->getK2(), cplx, Id() );
	ZeroOrder* r3 = z->makeHalfReaction( orig, enz->getK3(), cplx, Id() );

	unsigned int rateIndex = z->convertIdToReacIndex( orig->id() );
	if ( z->useOneWay_ ) {
		z->rates_[ rateIndex ] = r1;
		z->rates_[ rateIndex + 1 ] = r1;
		z->rates_[ rateIndex + 2 ] = r3;
	} else {
		z->rates_[ rateIndex ] = new BidirectionalReaction( r1, r2 );
		z->rates_[ rateIndex + 1 ] = r3;
	}

	vector< unsigned int > poolIndex;
	numReactants = r1->getReactants( poolIndex ); // Substrates
	for ( unsigned int i = 0; i < numReactants; ++i ) {
		int temp = z->N_.get( poolIndex[i], rateIndex );
		z->N_.set( poolIndex[i], rateIndex, temp - 1 );
	}
	numReactants = r2->getReactants( poolIndex );
	assert( numReactants == 1 ); // Should only be cplx as the only product
	unsigned int cplxPool = poolIndex[0];
	if ( z->useOneWay_ )
		z->N_.set( cplxPool, rateIndex + 1, 1 );
	else
		z->N_.set( cplxPool, rateIndex, 1 );

	// Now assign reaction 3. The complex is the only substrate here.
	unsigned int ritemp = z->useOneWay_ ? rateIndex + 2 : rateIndex + 1;
	z->N_.set( cplxPool, ritemp, -1 );
	// For the products, we go to the prd list directly.
	numReactants = orig->getOutputs( pools, prd ); 
	for ( unsigned int i = 0; i < numReactants; ++i ) {
		unsigned int j = z->convertIdToPoolIndex( pools[i] );
		int temp = z->N_.get( j, ritemp );
		z->N_.set( j, ritemp, temp + 1 );
	}
	// Enz is also a product here.
	unsigned int enzPool = z->convertIdToPoolIndex( enzId );
	z->N_.set( enzPool, ritemp, 1 );

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
