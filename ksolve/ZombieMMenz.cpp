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
			new EpFunc1< ZombieMMenz, ProcPtr >( &ZombieMMenz::eprocess ) );

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

	static Finfo* mmEnzFinfos[] = {
		&Km,	// Value
		&kcat,	// Value
		&process,			// DestFinfo
		&enzDest,				// DestFinfo
		&sub,				// SharedFinfo
		&prd,				// SharedFinfo
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

void ZombieMMenz::eprocess( Eref e, const Qinfo* q, ProcInfo* p )
{;}

void ZombieMMenz::process( const ProcInfo* p, const Eref& e )
{;}

void ZombieMMenz::reinit( const Eref& e, const Qinfo*q, ProcInfo* p )
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

/*
unsigned int  ZombieMMenz::convertId( Id id ) const
{
	unsigned int i = id.value() - objMapStart_;
	assert( i < objMap_.size() );
	i = objMap_[i];
	assert( i < rates_.size() );
	return i;
}

unsigned int ZombieMMenz::getSubIndex( Element* orig ) const
{
	static const SrcFinfo* sub = dynamic_cast< const SrcFinfo* >(
		MMenz::initCinfo()->findFinfo( "toSub" ) );
	assert( sub );
	const vector< MsgFuncBinding >* subVec = 
		orig->getMsgAndFunc( sub->getBindIndex() );

	if ( subVec->size() == 1 ) {
		const Msg* m = Msg::getMsg( (*subVec)[0].mid );
		assert( m );

		Id subId = m->e2()->id();
		if ( m->e2() == orig )
			subId = m->e1()->id();
		unsigned int i = subId.value() - objMapStart_;
		assert( i < objMap_.size() );
		unsigned int ret = objMap_[i];
		assert( ret < S_.size() );
		return ret;
	} else {
		cout << "Cannot handle multi-substrate MM enzymes just yet\n";
	}
	return 0;
}

unsigned int ZombieMMenz::getEnzIndex( Element* orig ) const
{
	static const DestFinfo* enz = dynamic_cast< const DestFinfo* >(
		MMenz::initCinfo()->findFinfo( "enz" ) );
	assert( enz );

	FuncId fid = enz->getFid();

	MsgId mid = orig->findCaller( fid );
	const Msg* m = Msg::getMsg( mid );
	assert( m );

	Id enzId = m->e1()->id();
	if ( m->e1() == orig )
		enzId = m->e2()->id();
	unsigned int i = enzId.value() - objMapStart_;
	assert( i < objMap_.size() );
	unsigned int ret = objMap_[i];
	assert( ret < S_.size() );
	return ret;
}
*/

// static func
void ZombieMMenz::zombify( Element* solver, Element* orig )
{
	static const DestFinfo* enz = dynamic_cast< const DestFinfo* >(
		MMenz::initCinfo()->findFinfo( "enz" ) );
	static const SrcFinfo* sub = dynamic_cast< const SrcFinfo* >(
		MMenz::initCinfo()->findFinfo( "toSub" ) );
	assert( enz );
	assert( sub );
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
