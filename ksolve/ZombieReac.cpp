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
#include "ZombieReac.h"
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
		new OpFunc1< ZombieReac, double >( &ZombieReac::sub ) );

static DestFinfo prd( "prdDest",
		"Handles # of molecules of product",
		new OpFunc1< ZombieReac, double >( &ZombieReac::prd ) );
	
static Finfo* subShared[] = {
	&toSub, &sub
};

static Finfo* prdShared[] = {
	&toPrd, &prd
};

const Cinfo* ZombieReac::initCinfo()
{
		//////////////////////////////////////////////////////////////
		// Field Definitions
		//////////////////////////////////////////////////////////////
		static ElementValueFinfo< ZombieReac, double > kf(
			"kf",
			"Forward rate constant",
			&ZombieReac::setKf,
			&ZombieReac::getKf
		);

		static ElementValueFinfo< ZombieReac, double > kb(
			"kb",
			"Backward rate constant",
			&ZombieReac::setKb,
			&ZombieReac::getKb
		);

		//////////////////////////////////////////////////////////////
		// MsgDest Definitions
		//////////////////////////////////////////////////////////////
		static DestFinfo process( "process",
			"Handles process call",
			new EpFunc1< ZombieReac, ProcPtr >( &ZombieReac::eprocess ) );

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

	static Finfo* zombieReacFinfos[] = {
		&kf,		// Value
		&kb,		// Value
		&process,	// DestFinfo
		&sub,		// SharedFinfo
		&prd,		// SharedFinfo
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
void ZombieReac::eprocess( Eref e, const Qinfo* q, ProcInfo* p )
{
}

void ZombieReac::process( const ProcInfo* p, const Eref& e )
{
}

void ZombieReac::sub( double v )
{
}

void ZombieReac::prd( double v )
{
}

//////////////////////////////////////////////////////////////
// Field Definitions
//////////////////////////////////////////////////////////////

void ZombieReac::setKf( Eref e, const Qinfo* q, double v )
{
	rates_[ convertId( e.id() ) ]->setR1( v );
}

double ZombieReac::getKf( Eref e, const Qinfo* q ) const
{
	return rates_[ convertId( e.id() ) ]->getR1();
}

void ZombieReac::setKb( Eref e, const Qinfo* q, double v )
{
	rates_[ convertId( e.id() ) ]->setR2( v );
}

double ZombieReac::getKb( Eref e, const Qinfo* q ) const
{
	return rates_[ convertId( e.id() ) ]->getR2();
}

//////////////////////////////////////////////////////////////
// Utility function
//////////////////////////////////////////////////////////////

unsigned int  ZombieReac::convertId( Id id ) const
{
	unsigned int i = id.value() - objMapStart_;
	assert( i < objMap_.size() );
	i = objMap_[i];
	assert( i < rates_.size() );
	return i;
}

unsigned int  ZombieReac::getReactants( 
	Element* orig, vector< unsigned int >& ret, bool isPrd ) const
{
	static const SrcFinfo* sub = dynamic_cast< const SrcFinfo* >(
		Reac::initCinfo()->findFinfo( "toSub" ) );
	static const SrcFinfo* prd = dynamic_cast< const SrcFinfo* >(
		Reac::initCinfo()->findFinfo( "toPrd" ) );
	assert( sub );
	assert( prd );
	const SrcFinfo* finfo = sub;
	if ( isPrd )
		finfo = prd;
	const vector< MsgFuncBinding >* msgVec = 
		orig->getMsgAndFunc( finfo->getBindIndex() );

	ret.resize( 0 );
	for ( unsigned int i = 0; i < msgVec->size(); ++i ) {
		const Msg* m = Msg::getMsg( (*msgVec)[i].mid );
		assert( m );
		Id id = m->e2()->id();
		if ( m->e2() == orig )
			id = m->e1()->id();
		unsigned int i = id.value() - objMapStart_;
		assert( i < objMap_.size() );
		unsigned int index = objMap_[i];
		assert( index < S_.size() );
		ret.push_back( index );
	}
	return ret.size();
}

ZeroOrder* ZombieReac::makeHalfReaction( 
	Element* orig, double rate, bool isReverse ) const
{
	vector< unsigned int > molIndex;
	unsigned int numReactants = 
		getReactants( orig, molIndex, isReverse ); 
	ZeroOrder* rateTerm;
	if ( numReactants == 1 ) {
		rateTerm = new FirstOrder( rate, &S_[ molIndex[0] ] );
	} else if ( numReactants == 2 ) {
		if ( molIndex[0] == molIndex[1] ) {
			rateTerm = new StochSecondOrderSingleSubstrate( rate,
				&S_[ molIndex[0] ] );
		} else {
			rateTerm = new SecondOrder( rate,
				&S_[ molIndex[0] ], &S_[ molIndex[1] ] );
		}
	} else if ( numReactants > 2 ) {
		vector< const double* > v;
		for ( unsigned int i = 0; i < numReactants; ++i ) {
			v.push_back( &S_[ molIndex[i] ] );
		}
		rateTerm = new NOrder( rate, v );
	}
	return rateTerm;
}

// static func
void ZombieReac::zombify( Element* solver, Element* orig )
{
	Element temp( orig->id(), zombieReacCinfo, solver->dataHandler() );
	Eref zer( &temp, 0 );
	Eref oer( orig, 0 );

	ZombieReac* z = reinterpret_cast< ZombieReac* >( zer.data() );
	Reac* reac = reinterpret_cast< Reac* >( oer.data() );

	ZeroOrder* forward = z->makeHalfReaction( orig, reac->getKf(), 0 );
	ZeroOrder* reverse = z->makeHalfReaction( orig, reac->getKb(), 1 );

	unsigned int rateIndex = z->convertId( orig->id() );
	z->rates_[ rateIndex ] = new BidirectionalReaction( forward, reverse );

	z->setKf( zer, 0, reac->getKf() );
	z->setKb( zer, 0, reac->getKb() );
	DataHandler* dh = new DataHandlerWrapper( solver->dataHandler() );
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
