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

/*
static SrcFinfo1< double > *requestSize() {
	static SrcFinfo1< double > requestSize( 
			"requestSize", 
			"Requests size (volume) in which reaction is embedded. Used for"
			"conversion to concentration units from molecule # units,"
			"and for calculations when resized."
			);
	return &requestSize;
}
*/

const Cinfo* ZombieReac::initCinfo()
{
		//////////////////////////////////////////////////////////////
		// Field Definitions
		//////////////////////////////////////////////////////////////
		static ElementValueFinfo< ZombieReac, double > kf(
			"kf",
			"Forward rate constant, in # units",
			&ZombieReac::setNumKf,
			&ZombieReac::getNumKf
		);

		static ElementValueFinfo< ZombieReac, double > kb(
			"kb",
			"Reverse rate constant, in # units",
			&ZombieReac::setNumKb,
			&ZombieReac::getNumKb
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

		static ReadOnlyElementValueFinfo< ZombieReac, unsigned int > numSub(
			"numSubstrates",
			"Number of substrates of reaction",
			&ZombieReac::getNumSub
		);

		static ReadOnlyElementValueFinfo< ZombieReac, unsigned int > numPrd(
			"numProducts",
			"Number of products of reaction",
			&ZombieReac::getNumPrd
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
		&numSub,		// ReadOnlyValue
		&numPrd,		// ReadOnlyValue
		//requestSize(),	// SrcFinfo
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

void ZombieReac::setNumKf( const Eref& e, const Qinfo* q, double v )
{
	cout << "Warning: numKf undefined for spatial reaction systems.\n";
	cout << "Use concKf if you want to change rates in this volume.\n";
	/*
	double volScale = 
		convertConcToNumRateUsingMesh( e, toSub(), 0 );
	concKf_ = v * volScale;
	rates_[ convertIdToReacIndex( e.id() ) ]->setR1( v );
	*/
}

double ZombieReac::getNumKf( const Eref& e, const Qinfo* q ) const
{
	// Return value for voxel 0. Conceivably I might want to use the
	// DataId part to specify which voxel to use, but that isn't in the
	// current definition for Reacs as being a single entity for the entire
	// compartment.
	return stoich_->getR1( stoich_->convertIdToReacIndex( e.id() ), 0 );
	// return rates_[ convertIdToReacIndex( e.id() ) ]->getR1();
}

void ZombieReac::setNumKb( const Eref& e, const Qinfo* q, double v )
{
	cout << "Warning: numKb undefined for spatial reaction systems.\n";
	cout << "Use concKb if you want to change rates in this volume.\n";
	/*
	if ( useOneWay_ )
		rates_[ convertIdToReacIndex( e.id() ) + 1 ]->setR1( v );
	else
		rates_[ convertIdToReacIndex( e.id() ) ]->setR2( v );
		*/
}

double ZombieReac::getNumKb( const Eref& e, const Qinfo* q ) const
{
	if ( stoich_->getOneWay() ) {
		return stoich_->getR1( stoich_->convertIdToReacIndex( e.id() ) + 1, 0 );
	} else {
		return stoich_->getR2( stoich_->convertIdToReacIndex( e.id() ), 0 );
	}
	/*
	if ( useOneWay_ )
		return rates_[ convertIdToReacIndex( e.id() ) + 1 ]->getR1();
	else
		return rates_[ convertIdToReacIndex( e.id() ) ]->getR2();
	*/
}

void ZombieReac::setConcKf( const Eref& e, const Qinfo* q, double v )
{
	concKf_ = v;
	stoich_->setReacKf( e, v );
	/*
	double volScale = 
		convertConcToNumRateUsingMesh( e, toSub(), 0 );
	rates_[ convertIdToReacIndex( e.id() ) ]->setR1( v / volScale );
	*/
	// setNumKf( e, q, v / volScale );
}

double ZombieReac::getConcKf( const Eref& e, const Qinfo* q ) const
{
	return concKf_;
	/*
	double volScale = 
		convertConcToNumRateUsingMesh( e, toSub(), 0 );
	return getNumKf( e, q ) * volScale;
	*/
}

void ZombieReac::setConcKb( const Eref& e, const Qinfo* q, double v )
{
	concKb_ = v;
	stoich_->setReacKb( e, v );
	/*
	double volScale = 
		convertConcToNumRateUsingMesh( e, toPrd(), 0 );
	// setNumKb( e, q, v / volScale );
	v /= volScale

	if ( useOneWay_ )
		rates_[ convertIdToReacIndex( e.id() ) + 1 ]->setR1( v );
	else
		rates_[ convertIdToReacIndex( e.id() ) ]->setR2( v );
		*/
}

double ZombieReac::getConcKb( const Eref& e, const Qinfo* q ) const
{
	return concKb_;
	/*
	double volScale = 
		convertConcToNumRateUsingMesh( e, toPrd(), 0 );
	return getNumKb( e, q ) * volScale;
	*/
}

unsigned int ZombieReac::getNumSub( const Eref& e, const Qinfo* q ) const
{
	const vector< MsgFuncBinding >* mfb =
	e.element()->getMsgAndFunc( toSub()->getBindIndex() );
	assert( mfb );
	return ( mfb->size() );
}

unsigned int ZombieReac::getNumPrd( const Eref& e, const Qinfo* q ) const
{
	const vector< MsgFuncBinding >* mfb =
	e.element()->getMsgAndFunc( toPrd()->getBindIndex() );
	assert( mfb );
	return ( mfb->size() );
}



//////////////////////////////////////////////////////////////
// Utility function
//////////////////////////////////////////////////////////////

ZeroOrder* ZombieReac::makeHalfReaction( 
	Element* orig, double rate, const SrcFinfo* finfo ) const
{
	vector< Id > mols;
	unsigned int numReactants = orig->getNeighbours( mols, finfo ); 
	ZeroOrder* rateTerm = 0;
	if ( numReactants == 1 ) {
		rateTerm = 
			new FirstOrder( rate, stoich_->convertIdToPoolIndex( mols[0] ) );
	} else if ( numReactants == 2 ) {
		rateTerm = new SecondOrder( rate,
				stoich_->convertIdToPoolIndex( mols[0] ), 
				stoich_->convertIdToPoolIndex( mols[1] ) );
	} else if ( numReactants > 2 ) {
		vector< unsigned int > v;
		for ( unsigned int i = 0; i < numReactants; ++i ) {
			v.push_back( stoich_->convertIdToPoolIndex( mols[i] ) );
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
	DataHandler* dh = orig->dataHandler()->copyUsingNewDinfo( ZombieReac::initCinfo()->dinfo() );

	Reac* reac = reinterpret_cast< Reac* >( orig->dataHandler()->data( 0 ));
	double concKf = reac->getConcKf( Eref( orig, 0 ), 0 );
	double concKb = reac->getConcKb( Eref( orig, 0 ), 0 );

	ZombieReac* zr = reinterpret_cast< ZombieReac* >( dh->data( 0 ) );
	zr->concKf_ = concKf;
	zr->concKb_ = concKb;
	zr->stoich_ = reinterpret_cast< Stoich* >( solver->dataHandler()->data( 0 ) );

	static const SrcFinfo* sub = dynamic_cast< const SrcFinfo* >(
		Reac::initCinfo()->findFinfo( "toSub" ) );
	static const SrcFinfo* prd = dynamic_cast< const SrcFinfo* >(
		Reac::initCinfo()->findFinfo( "toPrd" ) );
	
	assert( sub );
	assert( prd );

	ZeroOrder* forward = zr->makeHalfReaction( orig, concKf, sub );
	ZeroOrder* reverse = zr->makeHalfReaction( orig, concKb, prd );

	zr->stoich_->installReaction( forward, reverse, orig->id() );

	orig->zombieSwap( ZombieReac::initCinfo(), dh );

	zr->stoich_->setReacKf( Eref( orig, 0 ), concKf );
	zr->stoich_->setReacKb( Eref( orig, 0 ), concKb );
}

// Static func
void ZombieReac::unzombify( Element* zombie )
{
	DataHandler* dh = zombie->dataHandler()->copyUsingNewDinfo( Reac::initCinfo()->dinfo() );

	ZombieReac* zr = reinterpret_cast< ZombieReac* >( zombie->dataHandler()->data( 0 ));

	Reac* reac = reinterpret_cast< Reac* >( dh->data( 0 ));
	double concKf = zr->getConcKf( Eref( zombie, 0 ), 0 );
	double concKb = zr->getConcKb( Eref( zombie, 0 ), 0 );


	zombie->zombieSwap( Reac::initCinfo(), dh );
	// Now the zombie is a regular reac.

	reac->setConcKf( Eref( zombie, 0 ), 0, concKf );
	reac->setConcKb( Eref( zombie, 0 ), 0, concKb );
}
