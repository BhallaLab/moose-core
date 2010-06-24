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
	rates_[ convertIdToReacIndex( e.id() ) ]->setR1( v );
}

double ZombieEnz::getK1( Eref e, const Qinfo* q ) const
{
	return rates_[ convertIdToReacIndex( e.id() ) ]->getR1();
}

void ZombieEnz::setK2( Eref e, const Qinfo* q, double v )
{
	rates_[ convertIdToReacIndex( e.id() ) ]->setR2( v );
}

double ZombieEnz::getK2( Eref e, const Qinfo* q ) const
{
	return rates_[ convertIdToReacIndex( e.id() ) ]->getR2();
}

void ZombieEnz::setK3( Eref e, const Qinfo* q, double v )
{
	rates_[ convertIdToReacIndex( e.id() ) + 1 ]->setR1( v );
}

double ZombieEnz::getK3( Eref e, const Qinfo* q ) const
{
	return rates_[ convertIdToReacIndex( e.id() ) + 1 ]->getR1();
}

//////////////////////////////////////////////////////////////
// Utility function
//////////////////////////////////////////////////////////////

ZeroOrder* ZombieEnz::makeHalfReaction( 
	Element* orig, double rate, const SrcFinfo* finfo, Id enz ) const
{
	vector< Id > mols;
	unsigned int numReactants = orig->getOutputs( mols, finfo ); 
	if ( enz != Id() ) // Used to add the enz to the reactants.
		mols.push_back( enz );
	numReactants = mols.size();

	ZeroOrder* rateTerm;
	if ( numReactants == 1 ) {
		rateTerm = 
			new FirstOrder( rate, &S_[ convertIdToMolIndex( mols[0] ) ] );
	} else if ( numReactants == 2 ) {
		rateTerm = new SecondOrder( rate,
				&S_[ convertIdToMolIndex( mols[0] ) ], 
				&S_[ convertIdToMolIndex( mols[1] ) ] );
	} else if ( numReactants > 2 ) {
		vector< const double* > v;
		for ( unsigned int i = 0; i < numReactants; ++i ) {
			v.push_back( &S_[ convertIdToMolIndex( mols[i] ) ] );
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

	vector< Id > mols;
	unsigned int numReactants = orig->getOutputs( mols, enzFinfo ); 
	assert( numReactants == 1 );
	Id enzId = mols[0];

	ZeroOrder* r1 = z->makeHalfReaction( orig, enz->getK1(), sub, enzId );
	ZeroOrder* r2 = z->makeHalfReaction( orig, enz->getK2(), cplx, Id() );
	ZeroOrder* r3 = z->makeHalfReaction( orig, enz->getK3(), cplx, Id() );

	unsigned int rateIndex = z->convertIdToReacIndex( orig->id() );
	z->rates_[ rateIndex ] = new BidirectionalReaction( r1, r2 );
	z->rates_[ rateIndex + 1 ] = r3;

	vector< unsigned int > molIndex;
	numReactants = r1->getReactants( molIndex, z->S_ ); // Substrates
	for ( unsigned int i = 0; i < numReactants; ++i ) {
		int temp = z->N_.get( molIndex[i], rateIndex );
		z->N_.set( molIndex[i], rateIndex, temp - 1 );
	}
	numReactants = r2->getReactants( molIndex, z->S_ );
	assert( numReactants == 1 ); // Should only be cplx as the only product
	unsigned int cplxMol = molIndex[0];
	z->N_.set( cplxMol, rateIndex, 1 );

	// Now assign reaction 3. The complex is the only substrate here.
	z->N_.set( cplxMol, rateIndex + 1, -1 );
	// For the products, we go to the prd list directly.
	numReactants = orig->getOutputs( mols, prd ); 
	for ( unsigned int i = 0; i < numReactants; ++i ) {
		unsigned int j = z->convertIdToMolIndex( mols[i] );
		int temp = z->N_.get( j, rateIndex + 1 );
		z->N_.set( j, rateIndex + 1, temp - 1 );
	}
	// Enz is also a product here.
	unsigned int enzMol = z->convertIdToMolIndex( enzId );
	z->N_.set( enzMol, rateIndex + 1, 1 );

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
