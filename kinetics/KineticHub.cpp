/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2006 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/


#include "moose.h"
#include "RateTerm.h"
#include "KineticHub.h"
#include "Molecule.h"
#include "Reaction.h"
#include "Enzyme.h"
#include "ThisFinfo.h"
#include "SolveFinfo.h"

const Cinfo* initKineticHubCinfo()
{
	static TypeFuncPair processTypes[] =
	{
		TypeFuncPair( Ftype1< ProcInfo >::global(),
			RFCAST( &KineticHub::processFunc ) ),
		TypeFuncPair( Ftype1< ProcInfo >::global(),
			RFCAST( &KineticHub::reinitFunc ) ),
	};

	/**
	 * This is identical to the message sent from clock Ticks to
	 * objects. Here it is used to take over the Process message,
	 * usually only as a handle from the solver to the object.
	 * Here we are using the non-deprecated form.
	 */
	static Finfo* zombieShared[] =
	{
		new SrcFinfo( "process", Ftype1< ProcInfo >::global() ),
		new SrcFinfo( "reinit", Ftype1< ProcInfo >::global() ),
	};

	/**
	 * This is the destination of the several messages from the 
	 * Stoich object.
	 */
	static Finfo* hubShared[] =
	{
		new DestFinfo( "rateTermInfo", 
			Ftype2< vector< RateTerm* >*, bool >::global(),
			RFCAST( &KineticHub::rateTermFunc )
		),
		new DestFinfo( "rateSize", 
			Ftype3< unsigned int, unsigned int, unsigned int >::
			global(),
			RFCAST( &KineticHub::rateSizeFunc )
		),
		new DestFinfo( "molSize", 
			Ftype3< unsigned int, unsigned int, unsigned int >::
			global(),
			RFCAST( &KineticHub::molSizeFunc )
		),
		new DestFinfo( "molConnection",
			Ftype3< vector< double >* , 
				vector< double >* , 
				vector< Element *>*  
				>::global(),
			RFCAST( &KineticHub::molConnectionFunc )
		),
		new DestFinfo( "reacConnection",
			Ftype2< unsigned int, Element* >::global(),
			RFCAST( &KineticHub::reacConnectionFunc )
		),
		new DestFinfo( "enzConnection",
			Ftype2< unsigned int, Element* >::global(),
			RFCAST( &KineticHub::enzConnectionFunc )
		),
		new DestFinfo( "mmEnzConnection",
			Ftype2< unsigned int, Element* >::global(),
			RFCAST( &KineticHub::mmEnzConnectionFunc )
		),
	};

	static Finfo* kineticHubFinfos[] =
	{
	///////////////////////////////////////////////////////
	// Field definitions
	///////////////////////////////////////////////////////
		new ValueFinfo( "nMol", 
			ValueFtype1< unsigned int >::global(),
			GFCAST( &KineticHub::getNmol ), 
			&dummyFunc
		),
		new ValueFinfo( "nReac", 
			ValueFtype1< unsigned int >::global(),
			GFCAST( &KineticHub::getNreac ), 
			&dummyFunc
		),
		new ValueFinfo( "nEnz", 
			ValueFtype1< unsigned int >::global(),
			GFCAST( &KineticHub::getNenz ), 
			&dummyFunc
		),
	///////////////////////////////////////////////////////
	// MsgSrc definitions
	///////////////////////////////////////////////////////
	///////////////////////////////////////////////////////
	// MsgDest definitions
	///////////////////////////////////////////////////////
	///////////////////////////////////////////////////////
	// Shared definitions
	///////////////////////////////////////////////////////
		new SharedFinfo( "process", processTypes, 2 ),
		new SharedFinfo( "hub", hubShared, 
			      sizeof( hubShared ) / sizeof( Finfo* ) ),
		new SharedFinfo( "molSolve", zombieShared, 
			      sizeof( zombieShared ) / sizeof( Finfo* ) ),
		new SharedFinfo( "reacSolve", zombieShared, 
			      sizeof( zombieShared ) / sizeof( Finfo* ) ),
		new SharedFinfo( "enzSolve", zombieShared, 
			      sizeof( zombieShared ) / sizeof( Finfo* ) ),
		/*
		new SolveFinfo( "molSolve", molFields, 
			sizeof( molFields ) / sizeof( const Finfo* ) );
			*/
	};

	static Cinfo kineticHubCinfo(
		"KineticHub",
		"Upinder S. Bhalla, 2007, NCBS",
		"KineticHub: Object for controlling reaction systems on behalf of the\nStoich object. Interfaces both with the reaction system\n(molecules, reactions, enzymes\nand user defined rate terms) and also with the Stoich\nclass which generates the stoichiometry matrix and \nhandles the derivative calculations.",
		initNeutralCinfo(),
		kineticHubFinfos,
		sizeof(kineticHubFinfos )/sizeof(Finfo *),
		ValueFtype1< KineticHub >::global()
	);

	return &kineticHubCinfo;
}

static const Cinfo* kineticHubCinfo = initKineticHubCinfo();

/*
static const unsigned int reacSlot =
	initMoleculeCinfo()->getSlotIndex( "reac" );
static const unsigned int nSlot =
	initMoleculeCinfo()->getSlotIndex( "nSrc" );
*/

///////////////////////////////////////////////////
// Class function definitions
///////////////////////////////////////////////////

KineticHub::KineticHub()
	: S_( 0 ), Sinit_( 0 ), rates_( 0 ), 
	useHalfReacs_( 0 ), rebuildFlag_( 0 ),
	nMol_( 0 ), nBuf_( 0 ), nSumTot_( 0 )
{
	;
}

///////////////////////////////////////////////////
// Field function definitions
///////////////////////////////////////////////////

unsigned int KineticHub::getNmol( const Element* e )
{
	return static_cast< KineticHub* >( e->data() )->nMol_;
}

unsigned int KineticHub::getNreac( const Element* e )
{
	return static_cast< KineticHub* >( e->data() )->nMol_;
}

unsigned int KineticHub::getNenz( const Element* e )
{
	return static_cast< KineticHub* >( e->data() )->nMol_;
}

///////////////////////////////////////////////////
// Dest function definitions
///////////////////////////////////////////////////

void KineticHub::reinitFunc( const Conn& c, ProcInfo info )
{
	// Element* e = c.targetElement();
	// static_cast< KineticHub* >( e->data() )->processFuncLocal( e, info );
}

void KineticHub::processFunc( const Conn& c, ProcInfo info )
{
	// Element* e = c.targetElement();
	// static_cast< KineticHub* >( e->data() )->processFuncLocal( e, info );
}

void KineticHub::rateTermFunc( const Conn& c,
	vector< RateTerm* >* rates, bool useHalfReacs )
{
	KineticHub* kh = static_cast< KineticHub* >( c.data() );
	kh->rates_ = rates;
	kh->useHalfReacs_ = useHalfReacs;
}

/**
 * This sets up the groundwork for setting up the molecule connections.
 * We need to know how they are subdivided into mol, buf and sumTot to
 * assign the correct messages. The next call should come from the
 * molConnectionFunc which will provide the actual vectors.
 */
void KineticHub::molSizeFunc(  const Conn& c,
	unsigned int nMol, unsigned int nBuf, unsigned int nSumTot )
{
	static_cast< KineticHub* >( c.data() )->molSizeFuncLocal(
			nMol, nBuf, nSumTot );
}
void KineticHub::molSizeFuncLocal( 
		unsigned int nMol, unsigned int nBuf, unsigned int nSumTot )
{
	nMol_ = nMol;
	nBuf_ = nBuf;
	nSumTot_ = nSumTot;
}

void KineticHub::molConnectionFunc( const Conn& c,
	       	vector< double >*  S, vector< double >*  Sinit, 
		vector< Element *>*  elist )
{
	Element* e = c.targetElement();
	static_cast< KineticHub* >( e->data() )->
		molConnectionFuncLocal( e, S, Sinit, elist );
}

void KineticHub::molConnectionFuncLocal( Element* hub,
	       	vector< double >*  S, vector< double >*  Sinit, 
		vector< Element *>*  elist )
{
	static const Finfo* molSolveFinfo = 
		initKineticHubCinfo()->findFinfo( "molSolve" );
	// These fields will replace the original molecule fields so that
	// the lookups refer to the solver rather than the molecule.
	static Finfo* molFields[] =
	{
		new ValueFinfo( "n",
			ValueFtype1< double >::global(),
			GFCAST( &KineticHub::getMolN ),
			RFCAST( &KineticHub::setMolN )
		),
		new ValueFinfo( "nInit",
			ValueFtype1< double >::global(),
			GFCAST( &KineticHub::getMolNinit ),
			RFCAST( &KineticHub::setMolNinit )
		),
	};
	const ThisFinfo* tf = dynamic_cast< const ThisFinfo* >( 
		initMoleculeCinfo()->getThisFinfo( ) );
	assert( tf != 0 );
	static SolveFinfo molZombieFinfo( 
		molFields, 
		sizeof( molFields ) / sizeof( Finfo* ),
		tf
	);

	assert( nMol_ + nBuf_ + nSumTot_ == elist->size() );

	S_ = S;
	Sinit_ = Sinit;

	// cout << "in molConnectionFuncLocal\n";
	vector< Element* >::iterator i;
	// Note that here we have perfect alignment between the
	// order of the S_ and Sinit_ vectors and the elist vector.
	// This is used implicitly in the ordering of the process messages
	// that get set up between the Hub and the objects.
	for ( i = elist->begin(); i != elist->end(); i++ ) {
		zombify( hub, *i, molSolveFinfo, &molZombieFinfo );
	}
}

void KineticHub::rateSizeFunc(  const Conn& c,
	unsigned int nReac, unsigned int nEnz, unsigned int nMmEnz )
{
	static_cast< KineticHub* >( c.data() )->rateSizeFuncLocal(
		nReac, nEnz, nMmEnz );
}
void KineticHub::rateSizeFuncLocal( 
	unsigned int nReac, unsigned int nEnz, unsigned int nMmEnz )
{
	// Not sure what to do here.
	// cout << "in rateSizeFuncLocal\n";
}

void KineticHub::reacConnectionFunc( const Conn& c,
	unsigned int index, Element* reac )
{
	// cout << "in reacConnectionFunc for " << reac->name() << endl;
}

void KineticHub::enzConnectionFunc( const Conn& c,
	unsigned int index, Element* enz )
{
	cout << "in enzConnectionFunc for " << enz->name() << endl;
}

void KineticHub::mmEnzConnectionFunc( const Conn& c,
	unsigned int index, Element* mmEnz )
{
	cout << "in mmEnzConnectionFunc for " << mmEnz->name() << endl;
}

///////////////////////////////////////////////////
// Zombie object set/get function replacements
///////////////////////////////////////////////////

/**
 * Looks up the solver from the zombie element e. Returns the solver
 * element, or null on failure. Also passes back the index of the
 * zombie element, obtained by a simple lookup of the Conn index
 * connecting solver to zombie.
 */
KineticHub* getHubFromZombie( const Element* e, unsigned int& index )
{
	const SolveFinfo* f = dynamic_cast< const SolveFinfo* > (
			       	e->getThisFinfo() );
	if ( !f ) return 0;
	index = f->getIndex( e );
	return static_cast< KineticHub* >( f->getSolver( e ) );
}

void KineticHub::setMolN( const Conn& c, double value )
{
	unsigned int molIndex;
	KineticHub* kh = getHubFromZombie( c.targetElement(), molIndex );
	if ( kh && kh->S_ ) {
		assert ( molIndex < kh->S_->size() );
		( *kh->S_ )[molIndex] = value;
	}
	Molecule::setN( c, value );
}

double KineticHub::getMolN( const Element* e )
{
	unsigned int molIndex;
	KineticHub* kh = getHubFromZombie( e, molIndex );
	if ( kh && kh->S_ ) {
		assert ( molIndex < kh->S_->size() );
		return ( *kh->S_ )[molIndex];
	}
	return 0.0;
}

void KineticHub::setMolNinit( const Conn& c, double value )
{
	unsigned int molIndex;
	KineticHub* kh = getHubFromZombie( c.targetElement(), molIndex );
	if ( kh && kh->Sinit_ ) {
		assert ( molIndex < kh->Sinit_->size() );
		( *kh->Sinit_ )[molIndex] = value;
	}
	Molecule::setNinit( c, value );
}

double KineticHub::getMolNinit( const Element* e )
{
	unsigned int molIndex;
	KineticHub* kh = getHubFromZombie( e, molIndex );
	if ( kh && kh->Sinit_ ) {
		assert ( molIndex < kh->Sinit_->size() );
		return ( *kh->Sinit_ )[molIndex];
	}
	return 0.0;
}

/*
void KineticHubWrapper::reacConnectionFuncLocal( int rateTermIndex, Element* reac )
{
			Field reacSolve( this, "reacSolve" );
			zombify( reac, reacSolve );
			reacIndex_.push_back( rateTermIndex );
}
void KineticHubWrapper::enzConnectionFuncLocal( int rateTermIndex, Element* enz )
{
			Field enzSolve( this, "enzSolve" );
			zombify( enz, enzSolve );
			enzIndex_.push_back( rateTermIndex );
}
void KineticHubWrapper::mmEnzConnectionFuncLocal( int rateTermIndex, Element* enz )
{
			Field mmEnzSolve( this, "mmEnzSolve" );
			zombify( enz, mmEnzSolve );
			mmEnzIndex_.push_back( rateTermIndex );
}
void KineticHubWrapper::molFuncLocal( double n, double nInit, int mode, long index )
{
			if ( mode == SOLVER_GET ) {
				molSrc_.sendTo( index, (*S_)[index] );
			} else if ( mode == SOLVER_SET ) {
				(*S_)[index] = n;
				(*Sinit_)[index] = nInit;
			} else if ( mode == SOLVER_REBUILD ) {
				rebuildFlag_ = 1;
			}
}
void KineticHubWrapper::bufFuncLocal( double n, double nInit, int mode, long index )
{
			if ( mode == SOLVER_GET ) {
			} else if ( mode == SOLVER_SET ) {
				(*S_)[ index + nMol_ + nSumTot_ ] = 
					(*Sinit_)[ index + nMol_ + nSumTot_ ] = nInit;
			}
}
void KineticHubWrapper::sumTotFuncLocal( double n, double nInit, int mode, long index )
{
			if ( mode == SOLVER_GET ) {
				sumTotSrc_.sendTo( index, (*S_)[index + nMol_ ] );
			}
}
void KineticHubWrapper::reacFuncLocal( double kf, double kb, long index )
{
			unsigned long i = reacIndex_[ index ];
			if ( i >= 0 && i < rates_->size() - useOneWayReacs_ ) {
				if ( useOneWayReacs_ ) {
					( *rates_ )[i]->setRates( kf, 0 );
					( *rates_ )[i + 1]->setRates( kb, 0 );
				} else {
					( *rates_ )[i]->setRates( kf, kb );
				}
			}
}
void KineticHubWrapper::enzFuncLocal( double k1, double k2, double k3, long index )
{
			unsigned int i = enzIndex_[ index ];
			if ( i < rates_->size() - useOneWayReacs_ - 1) {
				if ( useOneWayReacs_ ) {
					( *rates_ )[i]->setRates( k1, 0 );
					( *rates_ )[i + 1]->setRates( k2, 0 );
					( *rates_ )[i + 2]->setRates( k3, 0 );
				} else {
					( *rates_ )[i]->setRates( k1, k2 );
					( *rates_ )[i + 1]->setRates( k3, 0 );
				}
			}
}
void KineticHubWrapper::mmEnzFuncLocal( double k1, double k2, double k3, long index )
{
			double Km = ( k2 + k3 ) / k1 ;
			unsigned int i = mmEnzIndex_[ index ];
			if ( i >= 0 && i < rates_->size() )
				( *rates_ )[i]->setRates( Km, k3 );
}
*/

///////////////////////////////////////////////////
// Other function definitions
///////////////////////////////////////////////////
void KineticHub::zombify( 
		Element* hub, Element* e, const Finfo* hubFinfo,
	       	Finfo* solveFinfo )
{
	// Replace the original procFinfo with one from the hub.
	const Finfo* procFinfo = e->findFinfo( "process" );
	procFinfo->dropAll( e );
	bool ret;
	ret = hubFinfo->add( hub, e, procFinfo );
	assert( ret );

	// Redirect original messages from the zombie to the hub.
	// Pending.

	// Replace the 'ThisFinfo' on the solved element
	e->setThisFinfo( solveFinfo );
}
