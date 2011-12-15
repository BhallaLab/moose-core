/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "SmolHeader.h"
#include "ElementValueFinfo.h"
#include "SmolEnz.h"
#include "Enz.h"
#include "DataHandlerWrapper.h"

const Cinfo* SmolEnz::initCinfo()
{
		//////////////////////////////////////////////////////////////
		// Field Definitions
		//////////////////////////////////////////////////////////////
		static ElementValueFinfo< SmolEnz, double > k1(
			"k1",
			"Forward rate constant",
			&SmolEnz::setK1,
			&SmolEnz::getK1
		);

		static ElementValueFinfo< SmolEnz, double > k2(
			"k2",
			"Forward rate constant",
			&SmolEnz::setK2,
			&SmolEnz::getK2
		);

		static ElementValueFinfo< SmolEnz, double > k3(
			"k3",
			"Forward rate constant",
			&SmolEnz::setK3,
			&SmolEnz::getK3
		);

		//////////////////////////////////////////////////////////////
		// MsgDest Definitions
		//////////////////////////////////////////////////////////////
		static DestFinfo process( "process",
			"Handles process call",
			new ProcOpFunc< SmolEnz >( &SmolEnz::process ) );

		static DestFinfo reinit( "reinit",
			"Handles reinit call",
			new ProcOpFunc< SmolEnz >( &SmolEnz::reinit ) );

		static DestFinfo group( "group",
			"Handle for group msgs. Doesn't do anything",
			new OpFuncDummy() );

		//////////////////////////////////////////////////////////////
		// Shared Msg Definitions
		//////////////////////////////////////////////////////////////

		static SrcFinfo2< double, double > toSub( 
				"toSub", 
				"Sends out increment of molecules on product each timestep"
				);
		static SrcFinfo2< double, double > toPrd( 
				"toPrd", 
				"Sends out increment of molecules on product each timestep"
				);
		static SrcFinfo2< double, double > toSmolEnz( 
				"toSmolEnz", 
				"Sends out increment of molecules on product each timestep"
				);
		static SrcFinfo2< double, double > toCplx( 
				"toCplx", 
				"Sends out increment of molecules on product each timestep"
				);
		static DestFinfo subDest( "subDest",
				"Handles # of molecules of substrate",
				new OpFunc1< SmolEnz, double >( &SmolEnz::dummy ) );
		static DestFinfo enzDest( "enzDest",
				"Handles # of molecules of SmolEnzyme",
				new OpFunc1< SmolEnz, double >( &SmolEnz::dummy ) );
		static DestFinfo prdDest( "prdDest",
				"Handles # of molecules of product. Dummy.",
				new OpFunc1< SmolEnz, double >( &SmolEnz::dummy ) );
		static DestFinfo cplxDest( "cplxDest",
				"Handles # of molecules of enz-sub complex",
				new OpFunc1< SmolEnz, double >( &SmolEnz::dummy ) );
		
		static Finfo* subShared[] = {
			&toSub, &subDest
		};
		static Finfo* enzShared[] = {
			&toSmolEnz, &enzDest
		};
		static Finfo* prdShared[] = {
			&toPrd, &prdDest
		};
		static Finfo* cplxShared[] = {
			&toCplx, &cplxDest
		};

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
		static Finfo* procShared[] = {
			&process, &reinit
		};
		static SharedFinfo proc( "proc",
			"Shared message for process and reinit",
			procShared, sizeof( procShared ) / sizeof( const Finfo* )
		);

	static Finfo* smolEnzFinfos[] = {
		&k1,	// Value
		&k2,	// Value
		&k3,	// Value
		&sub,				// SharedFinfo
		&prd,				// SharedFinfo
		&enz,				// SharedFinfo
		&cplx,				// SharedFinfo
		&proc,				// SharedFinfo
	};

	static Cinfo smolEnzCinfo (
		"SmolEnz",
		Neutral::initCinfo(),
		smolEnzFinfos,
		sizeof( smolEnzFinfos ) / sizeof ( Finfo* ),
		new Dinfo< SmolEnz >()
	);

	return &smolEnzCinfo;
}

 static const Cinfo* smolEnzCinfo = SmolEnz::initCinfo();

//////////////////////////////////////////////////////////////
// SmolEnz internal functions
//////////////////////////////////////////////////////////////

SmolEnz::SmolEnz( )
{ ; }

//////////////////////////////////////////////////////////////
// MsgDest Definitions
//////////////////////////////////////////////////////////////

void SmolEnz::dummy( double n ) // dummy
{;}

void SmolEnz::process( const Eref& e, ProcPtr p )
{;}

void SmolEnz::reinit( const Eref& e, ProcPtr p )
{;}

//////////////////////////////////////////////////////////////
// Field Definitions
//////////////////////////////////////////////////////////////

void SmolEnz::setK1( const Eref& e, const Qinfo* q, double v )
{
	// rates_[ convertIdToReacIndex( e.id() ) ]->setR1( v );
}

double SmolEnz::getK1( const Eref& e, const Qinfo* q ) const
{
	return 0;
	// return rates_[ convertIdToReacIndex( e.id() ) ]->getR1();
}

void SmolEnz::setK2( const Eref& e, const Qinfo* q, double v )
{
	// rates_[ convertIdToReacIndex( e.id() ) ]->setR2( v );
}

double SmolEnz::getK2( const Eref& e, const Qinfo* q ) const
{
	return 0;
	// return rates_[ convertIdToReacIndex( e.id() ) ]->getR2();
}

void SmolEnz::setK3( const Eref& e, const Qinfo* q, double v )
{
	// rates_[ convertIdToReacIndex( e.id() ) + 1 ]->setR1( v );
}

double SmolEnz::getK3( const Eref& e, const Qinfo* q ) const
{
	return 0;
	// return rates_[ convertIdToReacIndex( e.id() ) + 1 ]->getR1();
}

//////////////////////////////////////////////////////////////
// Utility function
//////////////////////////////////////////////////////////////

// static func
void SmolEnz::zombify( Element* solver, Element* orig )
{
/*
	static const SrcFinfo* sub = dynamic_cast< const SrcFinfo* >(
		Enz::initCinfo()->findFinfo( "toSub" ) );
	static const SrcFinfo* prd = dynamic_cast< const SrcFinfo* >(
		Enz::initCinfo()->findFinfo( "toPrd" ) );
	static const SrcFinfo* enzFinfo = dynamic_cast< const SrcFinfo* >(
		Enz::initCinfo()->findFinfo( "toEnz" ) );
	static const SrcFinfo* cplx = dynamic_cast< const SrcFinfo* >(
		Enz::initCinfo()->findFinfo( "toCplx" ) );

	Element temp( orig->id(), smolEnzCinfo, solver->dataHandler() );
	Eref zer( &temp, 0 );
	Eref oer( orig, 0 );

	SmolEnz* z = reinterpret_cast< SmolEnz* >( zer.data() );
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
	numReactants = r1->getReactants( molIndex ); // Substrates
	for ( unsigned int i = 0; i < numReactants; ++i ) {
		int temp = z->N_.get( molIndex[i], rateIndex );
		z->N_.set( molIndex[i], rateIndex, temp - 1 );
	}
	numReactants = r2->getReactants( molIndex );
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
		z->N_.set( j, rateIndex + 1, temp + 1 );
	}
	// Enz is also a product here.
	unsigned int enzMol = z->convertIdToMolIndex( enzId );
	z->N_.set( enzMol, rateIndex + 1, 1 );

	DataHandler* dh = new DataHandlerWrapper( solver->dataHandler(),
		orig->dataHandler() );
	orig->zombieSwap( smolEnzCinfo, dh );
	*/
}

// Static func
void SmolEnz::unzombify( Element* zombie )
{
/*
	Element temp( zombie->id(), zombie->cinfo(), zombie->dataHandler() );
	Eref zer( &temp, 0 );
	Eref oer( zombie, 0 );

	SmolEnz* z = reinterpret_cast< SmolEnz* >( zer.data() );

	// Here I am unsure how to recreate the correct kind of data handler
	// for the original. Do later.
	DataHandler* dh = 0;

	zombie->zombieSwap( Enz::initCinfo(), dh );

	Enz* m = reinterpret_cast< Enz* >( oer.data() );

	m->setK1( z->getK1( zer, 0 ) );
	m->setK2( z->getK2( zer, 0 ) );
	m->setK3( z->getK3( zer, 0 ) );
	*/
}
