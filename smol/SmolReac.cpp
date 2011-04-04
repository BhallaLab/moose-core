/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "SmolHeader.h"
#include "SmolReac.h"
#include "Reac.h"
#include "ElementValueFinfo.h"
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
		new OpFunc1< SmolReac, double >( &SmolReac::sub ) );

static DestFinfo prd( "prdDest",
		"Handles # of molecules of product",
		new OpFunc1< SmolReac, double >( &SmolReac::prd ) );
	
static Finfo* subShared[] = {
	&toSub, &sub
};

static Finfo* prdShared[] = {
	&toPrd, &prd
};

const Cinfo* SmolReac::initCinfo()
{
		//////////////////////////////////////////////////////////////
		// Field Definitions
		//////////////////////////////////////////////////////////////
		static ElementValueFinfo< SmolReac, double > kf(
			"kf",
			"Forward rate constant",
			&SmolReac::setKf,
			&SmolReac::getKf
		);

		static ElementValueFinfo< SmolReac, double > kb(
			"kb",
			"Backward rate constant",
			&SmolReac::setKb,
			&SmolReac::getKb
		);

		//////////////////////////////////////////////////////////////
		// MsgDest Definitions
		//////////////////////////////////////////////////////////////
		static DestFinfo process( "process",
			"Handles process call",
			new ProcOpFunc< SmolReac >( &SmolReac::process ) );

		static DestFinfo reinit( "reinit",
			"Handles reinit call",
			new ProcOpFunc< SmolReac >( &SmolReac::reinit ) );

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
		static Finfo* procShared[] = {
			&process, &reinit
		};
		static SharedFinfo proc( "proc",
			"Shared message for process and reinit",
			procShared, sizeof( procShared ) / sizeof( const Finfo* )
		);

	static Finfo* smolReacFinfos[] = {
		&kf,		// Value
		&kb,		// Value
		&sub,		// SharedFinfo
		&prd,		// SharedFinfo
		&proc,		// SharedFinfo
	};

	static Cinfo smolReacCinfo (
		"SmolReac",
		Neutral::initCinfo(),
		smolReacFinfos,
		sizeof( smolReacFinfos ) / sizeof ( Finfo* ),
		new Dinfo< SmolReac >()
	);

	return &smolReacCinfo;
}

//////////////////////////////////////////////////////////////
// Class definitions
//////////////////////////////////////////////////////////////
static const Cinfo* smolReacCinfo = SmolReac::initCinfo();

SmolReac::SmolReac()
{;}

SmolReac::~SmolReac()
{;}


//////////////////////////////////////////////////////////////
// MsgDest Definitions
//////////////////////////////////////////////////////////////

// Doesn't do anything on its own.
void SmolReac::process( const Eref& e, ProcPtr p )
{;}

void SmolReac::reinit( const Eref& e, ProcPtr p )
{;}


void SmolReac::sub( double v )
{
}

void SmolReac::prd( double v )
{
}

//////////////////////////////////////////////////////////////
// Field Definitions
//////////////////////////////////////////////////////////////

void SmolReac::setKf( const Eref& e, const Qinfo* q, double v )
{
//	rates_[ convertIdToReacIndex( e.id() ) ]->setR1( v );
}

double SmolReac::getKf( const Eref& e, const Qinfo* q ) const
{
//	return rates_[ convertIdToReacIndex( e.id() ) ]->getR1();
	return 0;
}

void SmolReac::setKb( const Eref& e, const Qinfo* q, double v )
{
	// rates_[ convertIdToReacIndex( e.id() ) ]->setR2( v );
}

double SmolReac::getKb( const Eref& e, const Qinfo* q ) const
{
	//return rates_[ convertIdToReacIndex( e.id() ) ]->getR2();
	return 0;
}

//////////////////////////////////////////////////////////////
// Utility function
//////////////////////////////////////////////////////////////

void buildHalfReac( simptr sim, const char* name,
	const vector< Id >& subs, const vector< Id >& prds,
	double rate )
{
	static const double EPSILON = 1e-9;

	if ( rate < EPSILON ) return;
	
	if ( subs.size() > 2 ) {
		cout << "Error: SmolReac::zombify: attempt to put > 2 substrates onto :" << name << endl;
		return;
	}
	const char* sub0Name = 0;
	if ( subs.size() > 0 )
		sub0Name = subs[0]()->getName().c_str();
	const char* sub1Name = 0;
	if ( subs.size() > 1 )
		sub1Name = subs[1]()->getName().c_str();

	const char** prdNames = new const char*[prds.size()];
	MolecState* prdStates = new MolecState[prds.size()];
	for ( unsigned int i = 0; i < prds.size(); ++i ) { 
		prdNames[i] = prds[i]()->getName().c_str();
		// Here I should check the state of the products. For now, solution.
		prdStates[i] = MSsoln;
	}

	ErrorCode ret = smolAddReaction( sim, name,
		sub0Name, MSsoln, sub1Name, MSsoln, 
		prds.size(), prdNames, prdStates );
	assert( ret == ECok );

	// Now to set the rates
	// I assume that the 'order' term specifies # of substrates.
	// I don't know what to do about the last arg, isinternal, even
	// after checking docs.
	ret = smolSetReactionRate( sim, subs.size(), name, rate, 0 );
	assert( ret == ECok );

	delete[] prdNames;
	delete[] prdStates;

}

// static func
void SmolReac::zombify( Element* solver, Element* orig )
{
	static const Finfo* toSub = orig->cinfo()->findFinfo( "toSub" );
	static const Finfo* toPrd = orig->cinfo()->findFinfo( "toPrd" );
	assert( toSub );
	assert( toPrd );

	Element temp( orig->id(), smolReacCinfo, solver->dataHandler() );
	Eref zer( &temp, 0 );
	Eref oer( orig, 0 );

	SmolReac* z = reinterpret_cast< SmolReac* >( zer.data() );
	Reac* r = reinterpret_cast< Reac* >( oer.data() );

	vector< Id > subs;
	vector< Id > prds;

	orig->getOutputs( subs, dynamic_cast< const SrcFinfo* >( toSub ) );
	orig->getOutputs( prds, dynamic_cast< const SrcFinfo* >( toPrd ) );

	string halfName = orig->getName() + "_f";
	buildHalfReac( z->sim_, halfName.c_str(), subs, prds, r->getKf() );

	halfName = orig->getName() + "_b";
	buildHalfReac( z->sim_, halfName.c_str(), prds, subs, r->getKb() );

	DataHandler* dh = new DataHandlerWrapper( solver->dataHandler() );
	orig->zombieSwap( smolReacCinfo, dh );
}

// Static func
void SmolReac::unzombify( Element* zombie )
{
	/*
	Element temp( zombie->id(), zombie->cinfo(), zombie->dataHandler() );
	Eref zer( &temp, 0 );
	Eref oer( zombie, 0 );

	SmolReac* z = reinterpret_cast< SmolReac* >( zer.data() );

	// Here I am unsure how to recreate the correct kind of data handler
	// for the original. Do later.
	DataHandler* dh = 0;

	zombie->zombieSwap( Reac::initCinfo(), dh );

	Reac* m = reinterpret_cast< Reac* >( oer.data() );

	m->setKf( z->getKf( zer, 0 ) );
	m->setKb( z->getKb( zer, 0 ) );
	*/
}
