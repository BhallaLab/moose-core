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

static SrcFinfo2< double, double > *toZombieEnz() {
	static SrcFinfo2< double, double > toZombieEnz( 
			"toZombieEnz", 
			"Sends out increment of molecules on product each timestep"
			);
	return &toZombieEnz;
}

static SrcFinfo2< double, double > *toCplx() {
	static SrcFinfo2< double, double > toCplx( 
			"toCplx", 
			"Sends out increment of molecules on product each timestep"
			);
	return &toCplx;
}

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

		static ElementValueFinfo< ZombieEnz, double > Km(
			"Km",
			"Michaelis-Menten constant, in conc (millimolar) units",
			&ZombieEnz::setKm,
			&ZombieEnz::getKm
		);

		static ElementValueFinfo< ZombieEnz, double > kcat(
			"kcat",
			"Forward rate constant, equivalent to k3",
			&ZombieEnz::setK3,
			&ZombieEnz::getK3
		);

		static ElementValueFinfo< ZombieEnz, double > ratio(
			"ratio",
			"Ratio of k2/k3",
			&ZombieEnz::setRatio,
			&ZombieEnz::getRatio
		);
		
		static ElementValueFinfo< ZombieEnz, double > concK1(
			"concK1",
			"K1 in conc units (1/millimolar.sec)",
			&ZombieEnz::setConcK1,
			&ZombieEnz::getConcK1
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

		static DestFinfo remesh( "remesh",
			"Compartment volume or meshing has changed, recompute rates",
			new EpFunc0< ZombieEnz >( &ZombieEnz::remesh ) );

		//////////////////////////////////////////////////////////////
		// Shared Msg Definitions
		//////////////////////////////////////////////////////////////

		static DestFinfo subDest( "subDest",
				"Handles # of molecules of substrate",
				new OpFunc1< ZombieEnz, double >( &ZombieEnz::dummy ) );
		static DestFinfo enzDest( "enzDest",
				"Handles # of molecules of ZombieEnzyme",
				new OpFunc1< ZombieEnz, double >( &ZombieEnz::dummy ) );
		static DestFinfo prdDest( "prdDest",
				"Handles # of molecules of product. Dummy.",
				new OpFunc1< ZombieEnz, double >( &ZombieEnz::dummy ) );
		static DestFinfo cplxDest( "cplxDest",
				"Handles # of molecules of enz-sub complex",
				new OpFunc1< ZombieEnz, double >( &ZombieEnz::dummy ) );
		static Finfo* subShared[] = {
			toSub(), &subDest
		};
		static Finfo* enzShared[] = {
			toZombieEnz(), &enzDest
		};
		static Finfo* prdShared[] = {
			toPrd(), &prdDest
		};
		static Finfo* cplxShared[] = {
			toCplx(), &cplxDest
		};
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
		&Km,	// Value
		&kcat,	// Value
		&ratio,	// Value
		&concK1,	// Value
		&sub,				// SharedFinfo
		&prd,				// SharedFinfo
		&enz,				// SharedFinfo
		&cplx,				// SharedFinfo
		&proc,				// SharedFinfo
		&remesh,			// DestFinfo
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


void ZombieEnz::remesh( const Eref& e, const Qinfo* q )
{   
	// setKm( e, q, Km_ );
}   


//////////////////////////////////////////////////////////////
// Field Definitions
//////////////////////////////////////////////////////////////

void ZombieEnz::setK1( const Eref& e, const Qinfo* q, double v )
{
	double volScale = 
		convertConcToNumRateUsingMesh( e, toSub(), 1 );

	concK1_ = v / volScale;
	stoich_->setEnzK1( e, concK1_ );
}

double ZombieEnz::getK1( const Eref& e, const Qinfo* q ) const
{
	return stoich_->getR1( stoich_->convertIdToReacIndex( e.id() ), 0 );
}

void ZombieEnz::setK2( const Eref& e, const Qinfo* q, double v )
{
	stoich_->setEnzK2( e, v );
}

double ZombieEnz::getK2( const Eref& e, const Qinfo* q ) const
{
	if ( stoich_->getOneWay() )
		return stoich_->getR1( 
			stoich_->convertIdToReacIndex( e.id() ) + 1, 0 );
	else
		return stoich_->getR2( stoich_->convertIdToReacIndex( e.id() ), 0 );
}

void ZombieEnz::setK3( const Eref& e, const Qinfo* q, double v )
{
	stoich_->setEnzK3( e, v );
}

double ZombieEnz::getK3( const Eref& e, const Qinfo* q ) const
{
	if ( stoich_->getOneWay() )
		return stoich_->getR1(
			stoich_->convertIdToReacIndex( e.id() ) + 2, 0 );
	else
		return stoich_->getR1(
			stoich_->convertIdToReacIndex( e.id() ) + 1, 0 );
}


void ZombieEnz::setKm( const Eref& e, const Qinfo* q, double v )
{
	double k2 = getK2( e, q );
	double k3 = getK3( e, q );
	stoich_->setEnzK1( e, ( k2 + k3 ) / concK1_ );
}

double ZombieEnz::getKm( const Eref& e, const Qinfo* q ) const
{
	double k2 = getK2( e, q );
	double k3 = getK3( e, q );

	return ( k2 + k3 ) / concK1_;
}

void ZombieEnz::setRatio( const Eref& e, const Qinfo* q, double v )
{
	double Km = getKm( e, q );
	double k2 = getK2( e, q );
	double k3 = getK3( e, q );

	k2 = v * k3;

	stoich_->setEnzK2( e, k2 );
	double k1 = ( k2 + k3 ) / Km;

	setConcK1( e, q, k1 );
}

double ZombieEnz::getRatio( const Eref& e, const Qinfo* q ) const
{
	double k2 = getK2( e, q );
	double k3 = getK3( e, q );
	return k2 / k3;
}

void ZombieEnz::setConcK1( const Eref& e, const Qinfo* q, double v )
{
	concK1_ = v;
	stoich_->setEnzK1( e, v );
	/*
	double volScale = 
		convertConcToNumRateUsingMesh( e, toSub(), 1 );

	double k1 = v * volScale;
	setK1( e, q, k1 );
	*/
}

double ZombieEnz::getConcK1( const Eref& e, const Qinfo* q ) const
{
	return concK1_;
}

//////////////////////////////////////////////////////////////
// Utility function
//////////////////////////////////////////////////////////////

ZeroOrder* ZombieEnz::makeHalfReaction( 
	Element* orig, double rate, const SrcFinfo* finfo, Id enz ) const
{
	vector< Id > pools;
	unsigned int numReactants = orig->getNeighbours( pools, finfo ); 
	if ( enz != Id() ) // Used to add the enz to the reactants.
		pools.push_back( enz );
	numReactants = pools.size();

	ZeroOrder* rateTerm = 0;
	if ( numReactants == 1 ) {
		rateTerm = 
			new FirstOrder( rate, stoich_->convertIdToPoolIndex( pools[0] ) );
	} else if ( numReactants == 2 ) {
		rateTerm = new SecondOrder( rate,
				stoich_->convertIdToPoolIndex( pools[0] ), 
				stoich_->convertIdToPoolIndex( pools[1] ) );
	} else if ( numReactants > 2 ) {
		vector< unsigned int > v;
		for ( unsigned int i = 0; i < numReactants; ++i ) {
			v.push_back( stoich_->convertIdToPoolIndex( pools[i] ) );
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

	assert( sub );
	assert( prd );
	assert( enzFinfo );
	assert( cplx );

	DataHandler* dh = orig->dataHandler()->copyUsingNewDinfo( 
		ZombieEnz::initCinfo()->dinfo() );
	
	ZombieEnz* z = reinterpret_cast< ZombieEnz* >( dh->data( 0 ) );
	z->stoich_ = reinterpret_cast< Stoich* >( solver->dataHandler()->data( 0 ) );
	Enz* enz = reinterpret_cast< Enz* >( orig->dataHandler()->data( 0 ) );
	Eref oer( orig, 0 );

	double concK1 = enz->getConcK1( oer, 0 );

	vector< Id > pools;
	unsigned int numReactants = orig->getNeighbours( pools, enzFinfo ); 
	assert( numReactants == 1 );
	Id enzId = pools[0];
	ZeroOrder* r1 = z->makeHalfReaction( orig, enz->getK1( oer, 0 ), sub, enzId );
	ZeroOrder* r2 = z->makeHalfReaction( orig, enz->getK2(), cplx, Id() );
	ZeroOrder* r3 = z->makeHalfReaction( orig, enz->getK3(), cplx, Id() );

	numReactants = orig->getNeighbours( pools, prd ); 
	z->stoich_->installEnzyme( r1, r2, r3, orig->id(), pools );

	orig->zombieSwap( ZombieEnz::initCinfo(), dh );
	z->concK1_ = concK1;
	z->stoich_->setEnzK1( Eref( orig, 0 ), concK1 );
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

	m->setK1( oer, 0, z->getK1( zer, 0 ) );
	m->setK2( z->getK2( zer, 0 ) );
	m->setK3( z->getK3( zer, 0 ) );
}
