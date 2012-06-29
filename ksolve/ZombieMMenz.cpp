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

static DestFinfo *enzDest() {
	static DestFinfo enzDest( "enzDest",
			"Handles # of molecules of ZombieMMenzyme",
			new OpFunc1< ZombieMMenz, double >( &ZombieMMenz::dummy ) );
	return &enzDest;
}


const Cinfo* ZombieMMenz::initCinfo()
{
		//////////////////////////////////////////////////////////////
		// Field Definitions
		//////////////////////////////////////////////////////////////
		static ElementValueFinfo< ZombieMMenz, double > Km(
			"Km",
			"Michaelis-Menten constant in SI conc units (millimolar)",
			&ZombieMMenz::setKm,
			&ZombieMMenz::getKm
		);

		static ElementValueFinfo< ZombieMMenz, double > numKm(
			"numKm",
			"Michaelis-Menten constant in number units, volume dependent",
			&ZombieMMenz::setNumKm,
			&ZombieMMenz::getNumKm
		);

		static ElementValueFinfo< ZombieMMenz, double > kcat(
			"kcat",
			"Forward rate constant for enzyme",
			&ZombieMMenz::setKcat,
			&ZombieMMenz::getKcat
		);

		static ReadOnlyElementValueFinfo< ZombieMMenz, unsigned int > numSub(
			"numSubstrates",
			"Number of substrates in this MM reaction. Usually 1."
			"Does not include the enzyme itself",
			&ZombieMMenz::getNumSub
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

		static DestFinfo remesh( "remesh",
		"Tells the ZombieMMEnz to recompute its numKm after remeshing",
		new EpFunc0< ZombieMMenz >( &ZombieMMenz::remesh ) );

		//////////////////////////////////////////////////////////////
		// Shared Msg Definitions
		//////////////////////////////////////////////////////////////
		static DestFinfo subDest( "subDest",
				"Handles # of molecules of substrate",
				new OpFunc1< ZombieMMenz, double >( &ZombieMMenz::dummy ) );
		static DestFinfo prdDest( "prdDest",
				"Handles # of molecules of product. Dummy.",
				new OpFunc1< ZombieMMenz, double >( &ZombieMMenz::dummy ) );
		static Finfo* subShared[] = {
			toSub(), &subDest
		};
		static Finfo* prdShared[] = {
			toPrd(), &prdDest
		};
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
		&numKm,	// Value
		&kcat,	// Value
		&numSub,	// ReadOnlyElementValue
		enzDest(),				// DestFinfo
		&sub,				// SharedFinfo
		&prd,				// SharedFinfo
		&proc,				// SharedFinfo
		&remesh,			// DestFinfo
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
	: Km_( 0.005 )
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

void ZombieMMenz::remesh( const Eref& e, const Qinfo* q )
{
	// cout << "ZombieMMenz::remesh for " << e << endl;
	stoich_->setMMenzKm( e, Km_ );
}

//////////////////////////////////////////////////////////////
// Field Definitions
//////////////////////////////////////////////////////////////

/*
double getEnzVol( const Eref& e )
{
	vector< Id > enzMol;
	e.element()->getNeighbours( enzMol, enzDest() );
	assert( enzMol.size() == 1 );
	const Finfo* f1 = enzMol[0].element()->cinfo()->findFinfo( "requestSize" );
	const SrcFinfo* sf = dynamic_cast< const SrcFinfo* >( f1 );
	assert( sf );
	double vol = lookupSizeFromMesh( enzMol[0].eref(), sf );
	assert( vol > 0.0 );
	return vol;
}
*/

void ZombieMMenz::setKm( const Eref& e, const Qinfo* q, double v )
{
	Km_ = v;
	stoich_->setMMenzKm( e, v );
	/*
	double volScale = convertConcToNumRateUsingMesh( e, toSub(), 1 );
	// double numKm = v * NA * CONC_UNIT_CONV * getEnzVol( e );

	// First rate is Km
	rates_[ convertIdToPoolIndex( e.id() ) ]->setR1( v * volScale ); 
	*/
}

double ZombieMMenz::getKm( const Eref& e, const Qinfo* q ) const
{
	return Km_;
}

void ZombieMMenz::setNumKm( const Eref& e, const Qinfo* q, double v )
{
	double volScale = convertConcToNumRateUsingMesh( e, toSub(), 1 );
	Km_ = v / volScale;
	setKm( e, q, Km_ );
	// rates_[ convertIdToPoolIndex( e.id() ) ]->setR1( v ); 
}

double ZombieMMenz::getNumKm( const Eref& e, const Qinfo* q ) const
{
	double numKm = stoich_->getR1( stoich_->convertIdToPoolIndex( e.id() ), 0 );
	
	return numKm;
}

void ZombieMMenz::setKcat( const Eref& e, const Qinfo* q, double v )
{
	stoich_->setMMenzKcat( e, v );
}

double ZombieMMenz::getKcat( const Eref& e, const Qinfo* q ) const
{
	// Second rate is kcat
	double kcat = stoich_->getR2( stoich_->convertIdToPoolIndex( e.id() ), 0 );
	return kcat;
}

unsigned int ZombieMMenz::getNumSub( const Eref& e, const Qinfo* q ) const
{
	const vector< MsgFuncBinding >* mfb =
		e.element()->getMsgAndFunc( toSub()->getBindIndex() );
	assert( mfb );
	return ( mfb->size() );
}

//////////////////////////////////////////////////////////////
// Utility function
//////////////////////////////////////////////////////////////

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

	DataHandler* dh = orig->dataHandler()->copyUsingNewDinfo( 
		ZombieMMenz::initCinfo()->dinfo() );
	MMenz* mmEnz = reinterpret_cast< MMenz* >( 
		orig->dataHandler()->data( 0 ) );

	Eref oer( orig, 0 );
	double Km = mmEnz->getKm( oer, 0 );

	ZombieMMenz* z = reinterpret_cast< ZombieMMenz* >( dh->data( 0 ) );
	z->Km_ = Km;
	z->stoich_ = reinterpret_cast< Stoich* >( 	
		solver->dataHandler()->data( 0 ) );


	/// Now set up the RateTerm
	vector< Id > subvec;
	vector< Id > prdvec;
	unsigned int rateIndex = z->stoich_->convertIdToReacIndex( orig->id() );
	unsigned int num = orig->getNeighbours( subvec, enz );
	unsigned int enzIndex = z->stoich_->convertIdToPoolIndex( subvec[0] );
	MMEnzymeBase* meb;

	num = orig->getNeighbours( subvec, sub );
	if ( num == 1 ) {
		unsigned int subIndex = z->stoich_->convertIdToPoolIndex( subvec[0] );
		meb = new MMEnzyme1( mmEnz->getNumKm( oer, 0 ), mmEnz->getKcat(),
			enzIndex, subIndex );
	} else if ( num > 1 ) {
		vector< unsigned int > v;
		for ( unsigned int i = 0; i < num; ++i )
			v.push_back( z->stoich_->convertIdToPoolIndex( subvec[i] ) );
		ZeroOrder* rateTerm = new NOrder( 1.0, v );
		meb = new MMEnzyme( mmEnz->getNumKm( oer, 0 ), mmEnz->getKcat(),
			enzIndex, rateTerm );
	} else {
		cout << "Error: ZombieMMenz::zombify: No substrates for "  <<
			orig->id().path() << endl;
		cout << "Will ignore and continue, but don't be surprised if "
		"simulation fails.\n";
		// assert( 0 );
		delete dh;
		return;
	}

	num = orig->getNeighbours( prdvec, prd );

	z->stoich_->installMMenz( meb, rateIndex, subvec, prdvec );

	orig->zombieSwap( ZombieMMenz::initCinfo(), dh );
	z->stoich_->setMMenzKm( Eref( orig, 0 ), Km );
}

/*
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
	unsigned int num = orig->getNeighbours( pools, enz );
	unsigned int enzIndex = z->convertIdToPoolIndex( pools[0] );

	num = orig->getNeighbours( pools, sub );
	if ( num == 1 ) {
		unsigned int subIndex = z->convertIdToPoolIndex( pools[0] );
		assert( num == 1 );
		z->rates_[ rateIndex ] = new MMEnzyme1( 
			mmEnz->getNumKm( oer, 0 ), mmEnz->getKcat(),
			enzIndex, subIndex );
	} else if ( num > 1 ) {
		vector< unsigned int > v;
		for ( unsigned int i = 0; i < num; ++i )
			v.push_back( z->convertIdToPoolIndex( pools[i] ) );
		ZeroOrder* rateTerm = new NOrder( 1.0, v );
		z->rates_[ rateIndex ] = new MMEnzyme( 
			mmEnz->getNumKm( oer, 0 ), mmEnz->getKcat(),
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
	num = orig->getNeighbours( pools, prd );
	for ( unsigned int i = 0; i < num; ++i ) {
		unsigned int poolIndex = z->convertIdToPoolIndex( pools[i] );
		int temp = z->N_.get( poolIndex, rateIndex );
		z->N_.set( poolIndex, rateIndex, temp + 1 );
	}

	DataHandler* dh = new DataHandlerWrapper( solver->dataHandler(),
		orig->dataHandler() );
	orig->zombieSwap( zombieMMenzCinfo, dh );
}
*/

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

	m->setKm( oer, 0, z->getNumKm( zer, 0 ) );
	m->setKcat( z->getKcat( zer, 0 ) );
}
