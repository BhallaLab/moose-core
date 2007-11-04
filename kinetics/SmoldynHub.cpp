/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/


#include "moose.h"
#include "../element/Neutral.h"
#include "RateTerm.h"
#include "SmoldynHub.h"
#include "Molecule.h"
#include "Particle.h"
#include "Reaction.h"
#include "Enzyme.h"
#include "ThisFinfo.h"
#include "SolveFinfo.h"

#include "Smoldyn/source/smollib.h"
// #include "../element/Wildcard.h"


const Cinfo* initSmoldynHubCinfo()
{
	static Finfo* processShared[] =
	{
		new DestFinfo( "process", Ftype1< ProcInfo >::global(),
			RFCAST( &SmoldynHub::processFunc ) ),
		new DestFinfo( "reinit", Ftype1< ProcInfo >::global(),
			RFCAST( &SmoldynHub::reinitFunc ) ),
	};
	static Finfo* process = new SharedFinfo( "process", processShared,
		sizeof( processShared ) / sizeof( Finfo* ) );

	/**
	 * Takes over the process message to each of the kinetic objects.
	 * Replaces the original message usually sent by the clock Ticks.
	 */
	static Finfo* zombieShared[] =
	{
		new SrcFinfo( "process", Ftype1< ProcInfo >::global() ),
		new SrcFinfo( "reinit", Ftype1< ProcInfo >::global() ),
	};

	
	/**
	 * Handles reaction structure info from the Stoich object
	 */
	static Finfo* hubShared[] =
	{
		new DestFinfo( "rateTermInfo",
			Ftype2< vector< RateTerm* >*, bool >::global(),
			RFCAST( &SmoldynHub::rateTermFunc )
		),
		new DestFinfo( "rateSize", 
			Ftype3< unsigned int, unsigned int, unsigned int >::
			global(),
			RFCAST( &SmoldynHub::rateSizeFunc )
		),
		new DestFinfo( "molSize", 
			Ftype3< unsigned int, unsigned int, unsigned int >::
			global(),
			RFCAST( &SmoldynHub::molSizeFunc )
		),
		new DestFinfo( "molConnection",
			Ftype3< vector< double >* , 
				vector< double >* , 
				vector< Element *>*  
				>::global(),
			RFCAST( &SmoldynHub::molConnectionFunc )
		),
		new DestFinfo( "reacConnection",
			Ftype2< unsigned int, Element* >::global(),
			RFCAST( &SmoldynHub::reacConnectionFunc )
		),
		new DestFinfo( "enzConnection",
			Ftype2< unsigned int, Element* >::global(),
			RFCAST( &SmoldynHub::enzConnectionFunc )
		),
		new DestFinfo( "mmEnzConnection",
			Ftype2< unsigned int, Element* >::global(),
			RFCAST( &SmoldynHub::mmEnzConnectionFunc )
		),
		new DestFinfo( "clear",
			Ftype0::global(),
			RFCAST( &SmoldynHub::clearFunc )
		),
	};

	static Finfo* smoldynHubFinfos[] =
	{
	///////////////////////////////////////////////////////
	// Field definitions
	///////////////////////////////////////////////////////
		new ValueFinfo( "nMol", 
			ValueFtype1< unsigned int >::global(),
			GFCAST( &SmoldynHub::getNspecies ), 
			&dummyFunc
		),
		new ValueFinfo( "nReac", 
			ValueFtype1< unsigned int >::global(),
			GFCAST( &SmoldynHub::getNreac ), 
			&dummyFunc
		),
		new ValueFinfo( "nEnz", 
			ValueFtype1< unsigned int >::global(),
			GFCAST( &SmoldynHub::getNenz ), 
			&dummyFunc
		),
		/*
		new ValueFinfo( "path", 
			ValueFtype1< string >::global(),
			GFCAST( &SmoldynHub::getPath ),
			RFCAST( &SmoldynHub::setPath )
		),
		*/
	///////////////////////////////////////////////////////
	// MsgSrc definitions
	///////////////////////////////////////////////////////
	//	new SrcFinfo( "nSrc", Ftype1< double >::global() ),
	///////////////////////////////////////////////////////
	// MsgDest definitions
	///////////////////////////////////////////////////////
		new DestFinfo( "destroy", Ftype0::global(),
			&SmoldynHub::destroy ),
		new DestFinfo( "molSum", Ftype1< double >::global(),
			RFCAST( &SmoldynHub::molSum ) ),
		// override the Neutral::childFunc here, so that when this
		// is deleted all the zombies are reanimated.
		new DestFinfo( "child", Ftype1< int >::global(),
			RFCAST( &SmoldynHub::childFunc ) ),

	///////////////////////////////////////////////////////
	// Synapse definitions
	///////////////////////////////////////////////////////
	///////////////////////////////////////////////////////
	// Shared definitions
	///////////////////////////////////////////////////////
		process,
		new SharedFinfo( "hub", hubShared, 
			      sizeof( hubShared ) / sizeof( Finfo* ) ),
		new SharedFinfo( "molSolve", zombieShared, 
			      sizeof( zombieShared ) / sizeof( Finfo* ) ),
		new SharedFinfo( "reacSolve", zombieShared, 
			      sizeof( zombieShared ) / sizeof( Finfo* ) ),
		new SharedFinfo( "enzSolve", zombieShared, 
			      sizeof( zombieShared ) / sizeof( Finfo* ) ),
		new SharedFinfo( "mmEnzSolve", zombieShared, 
			      sizeof( zombieShared ) / sizeof( Finfo* ) ),
	};

	// Schedule smoldynHubs for the slower clock, stage 0.
	static SchedInfo schedInfo[] = { { process, 0, 0 } };
	
	static Cinfo smoldynHubCinfo(
		"SmoldynHub",
		"Upinder S. Bhalla, 2007, NCBS",
		"SmoldynHub: Interface object between Smoldyn (by Steven Andrews) and MOOSE.",
		initNeutralCinfo(),
		smoldynHubFinfos,
		sizeof( smoldynHubFinfos )/sizeof(Finfo *),
		ValueFtype1< SmoldynHub >::global(),
			schedInfo, 1
	);

	return &smoldynHubCinfo;
}

static const Cinfo* smoldynHubCinfo = initSmoldynHubCinfo();

/*
const Finfo* SmoldynHub::particleFinfo = 
	initSmoldynHubCinfo()->findFinfo( "particleFinfo" );
	*/

const Finfo* SmoldynHub::molSolveFinfo =
	initSmoldynHubCinfo()->findFinfo( "molSolve" );

static const Finfo* molSumFinfo =
	initSmoldynHubCinfo()->findFinfo( "molSum" );

static const Finfo* reacSolveFinfo = 
	initSmoldynHubCinfo()->findFinfo( "reacSolve" );
static const Finfo* enzSolveFinfo = 
	initSmoldynHubCinfo()->findFinfo( "enzSolve" );
static const Finfo* mmEnzSolveFinfo = 
	initSmoldynHubCinfo()->findFinfo( "mmEnzSolve" );

static const unsigned int molSumSlot =
	initSmoldynHubCinfo()->getSlotIndex( "molSum" );

/*
static const unsigned int reacSlot =
	initSmoldynHubCinfo()->getSlotIndex( "reac.n" );
static const unsigned int nSlot =
	initSmoldynHubCinfo()->getSlotIndex( "nSrc" );
*/

void redirectDestMessages(
	Element* hub, Element* e, const Finfo* hubFinfo, const Finfo* eFinfo,
	unsigned int eIndex, vector< unsigned int >& map,
	vector< Element* >* elist, bool retain = 0 );

void redirectDynamicMessages( Element* e );
void unzombify( const Conn& c );

///////////////////////////////////////////////////
// Class function definitions
///////////////////////////////////////////////////

SmoldynHub::SmoldynHub()
	: simptr_( 0 )
{
		;
}

void SmoldynHub::destroy( const Conn& c )
{
	clearFunc( c );
	Neutral::destroy( c );
}

void SmoldynHub::childFunc( const Conn& c, int stage )
{
	if ( stage == 1 ) // Clean out zombies before deleting messages.
		clearFunc( c );
	Neutral::destroy( c );
}

/**
 * Here we add external inputs to a molecule. This message replaces
 * the SumTot input to a molecule when it is used for controlling its
 * number. It is not meant as a substitute for SumTot between molecules.
 * It is a bit tricky in Smoldyn, because we cannot just add numbers, but
 * have to give them a position.
 */
void SmoldynHub::molSum( const Conn& c, double val )
{
	Element* hub = c.targetElement();
	unsigned int index = hub->connDestRelativeIndex( c, molSumSlot );
	SmoldynHub* sh = static_cast< SmoldynHub* >( hub->data() );

	assert( index < sh->molSumMap_.size() );
	// Do something intelligent here with the molSum input.
}

///////////////////////////////////////////////////
// Hub utility function definitions
///////////////////////////////////////////////////

/**
 * Looks up the solver from the zombie element e. Returns the solver
 * element, or null on failure. 
 * It needs the originating Finfo on the solver that connects to the zombie,
 * as the srcFinfo.
 * Also passes back the index of the zombie element on this set of
 * messages. This is NOT the absolute Conn index.
 */
SmoldynHub* SmoldynHub::getHubFromZombie( 
	const Element* e, const Finfo* srcFinfo, unsigned int& index )
{
	const SolveFinfo* f = dynamic_cast< const SolveFinfo* > (
			       	e->getThisFinfo() );
	if ( !f ) return 0;
	const Conn& c = f->getSolvedConn( e );
	unsigned int slot;
       	srcFinfo->getSlotIndex( srcFinfo->name(), slot );
	Element* hub = c.targetElement();
	index = hub->connSrcRelativeIndex( c, slot );
	return static_cast< SmoldynHub* >( hub->data() );
}

void SmoldynHub::setPos( unsigned int molIndex, double value, 
			unsigned int i, unsigned int dim )
{
}

double SmoldynHub::getPos( unsigned int molIndex, unsigned int i, 
			unsigned int dim )
{
	return 0.0;
}

void SmoldynHub::setPosVector( unsigned int molIndex, 
			const vector< double >& value, unsigned int dim )
{
}

void SmoldynHub::getPosVector( unsigned int molIndex,
			vector< double >& value, unsigned int dim )
{
}

void SmoldynHub::setNinit( unsigned int molIndex, unsigned int value )
{
}

unsigned int SmoldynHub::getNinit( unsigned int molIndex )
{
	return 0;
}

void SmoldynHub::setNmol( unsigned int molIndex, unsigned int value )
{
	;
}

unsigned int SmoldynHub::getNmol( unsigned int molIndex )
{
	return 0;
}

void SmoldynHub::setD( unsigned int molIndex, double value )
{
}

double SmoldynHub::getD( unsigned int molIndex )
{
	return 0.0;
}


///////////////////////////////////////////////////
// Field function definitions
///////////////////////////////////////////////////
unsigned int SmoldynHub::getNspecies( const Element* e )
{
	return static_cast< SmoldynHub* >( e->data() )->numSpecies();
}

unsigned int SmoldynHub::numSpecies() const
{
	return 0;
}

unsigned int SmoldynHub::getNreac( const Element* e )
{
	return static_cast< SmoldynHub* >( e->data() )->numReac();
}

unsigned int SmoldynHub::numReac() const
{
	return 0;
}

unsigned int SmoldynHub::getNenz( const Element* e )
{
	return static_cast< SmoldynHub* >( e->data() )->numEnz();
}

unsigned int SmoldynHub::numEnz() const
{
	return 0;
}

string SmoldynHub::getPath( const Element* e )
{
	//return static_cast< const SmoldynHub* >( e->data() )->path_;
	return "";
}

void SmoldynHub::setPath( const Conn& c, string value )
{
	Element* e = c.targetElement();
	static_cast< SmoldynHub* >( e->data() )->localSetPath( e, value );
}

///////////////////////////////////////////////////
// Dest function definitions
///////////////////////////////////////////////////

void SmoldynHub::reinitFunc( const Conn& c, ProcInfo info )
{
	static_cast< SmoldynHub* >( c.data() )->reinitFuncLocal( 
					c.targetElement() );
}

void SmoldynHub::reinitFuncLocal( Element* e )
{
	// do stuff here.
}

void SmoldynHub::processFunc( const Conn& c, ProcInfo info )
{
	Element* e = c.targetElement();
	static_cast< SmoldynHub* >( e->data() )->processFuncLocal( e, info );
}

void SmoldynHub::processFuncLocal( Element* e, ProcInfo info )
{
	// do stuff here
}

///////////////////////////////////////////////////
// This may go soon
///////////////////////////////////////////////////

void SmoldynHub::localSetPath( Element* stoich, const string& value )
{
	/*
	path_ = value;
	vector< Element* > ret;
	wildcardFind( path_, ret );
	if ( ret.size() > 0 ) {
		;
	}
	cout << "found " << ret.size() << " elements\n";
	*/
}



///////////////////////////////////////////////////
// This is where the reaction system from MOOSE is zombified and 
// incorporated into the solution engine.
///////////////////////////////////////////////////

/**
 * This sets up the local version of the rate term array, used to 
 * figure out reaction structures.
 */
void SmoldynHub::rateTermFunc( const Conn& c,
	vector< RateTerm* >* rates, bool useHalfReacs )
{
	// the useHalfReacs flag is irrelevant here, since Smoldyn always 
	// considers irreversible reactions
	SmoldynHub* sh = static_cast< SmoldynHub* >( c.data() );
	sh->rates_ = rates;
}

/**
 * This sets up the groundwork for setting up the molecule connections.
 * We need to know how they are subdivided into mol, buf and sumTot to
 * assign the correct messages. The next call should come from the
 * molConnectionFunc which will provide the actual vectors.
 */
void SmoldynHub::molSizeFunc(  const Conn& c,
	unsigned int nMol, unsigned int nBuf, unsigned int nSumTot )
{
	static_cast< SmoldynHub* >( c.data() )->molSizeFuncLocal(
			nMol, nBuf, nSumTot );
}
void SmoldynHub::molSizeFuncLocal( 
		unsigned int nMol, unsigned int nBuf, unsigned int nSumTot )
{
	nMol_ = nMol;
	nBuf_ = nBuf;
	nSumTot_ = nSumTot;
}

/**
 * This function zombifies the molecules
 */
void SmoldynHub::molConnectionFunc( const Conn& c,
	       	vector< double >*  S, vector< double >*  Sinit, 
		vector< Element *>*  elist )
{
	Element* e = c.targetElement();
	static_cast< SmoldynHub* >( e->data() )->
		molConnectionFuncLocal( e, S, Sinit, elist );
}

void SmoldynHub::molConnectionFuncLocal( Element* hub,
	       	vector< double >*  S, vector< double >*  Sinit, 
		vector< Element *>*  elist )
{
	assert( nMol_ + nBuf_ + nSumTot_ == elist->size() );

	S_ = S;
	Sinit_ = Sinit;

	// cout << "in molConnectionFuncLocal\n";
	vector< Element* >::iterator i;
	// Note that here we have perfect alignment between the
	// order of the S_ and Sinit_ vectors and the elist vector.
	// This is used implicitly in the ordering of the process messages
	// that get set up between the Hub and the objects.
	const Finfo* sumTotFinfo = initParticleCinfo()->findFinfo( "sumTotal" );
	Finfo* particleFinfo = 
		const_cast< Finfo* >( initParticleCinfo()->getThisFinfo() );

	for ( i = elist->begin(); i != elist->end(); i++ ) {
		zombify( hub, *i, molSolveFinfo, particleFinfo );
		redirectDynamicMessages( *i );
	}
	// Here we should really set up a 'set' of mols to check if the
	// sumTotMessage is coming from in or outside the tree.
	// Since I'm hazy about the syntax, here I'm just using the elist.
	for ( i = elist->begin(); i != elist->end(); i++ ) {
		// Here we replace the sumTotMessages from outside the tree.
		redirectDestMessages( hub, *i, molSumFinfo, sumTotFinfo, 
			i - elist->begin(), molSumMap_, elist, 1 );
	}
}

void SmoldynHub::rateSizeFunc(  const Conn& c,
	unsigned int nReac, unsigned int nEnz, unsigned int nMmEnz )
{
	static_cast< SmoldynHub* >( c.data() )->rateSizeFuncLocal(
		c.targetElement(), nReac, nEnz, nMmEnz );
}
void SmoldynHub::rateSizeFuncLocal( Element* hub, 
	unsigned int nReac, unsigned int nEnz, unsigned int nMmEnz )
{
	// Ensure we have enough space allocated in each of the maps
	reacMap_.resize( nReac );
	enzMap_.resize( nEnz );
	mmEnzMap_.resize( nMmEnz );

	// Not sure what to do here.
	// cout << "in rateSizeFuncLocal\n";
}

void SmoldynHub::reacConnectionFunc( const Conn& c,
	unsigned int index, Element* reac )
{
	Element* e = c.targetElement();
	static_cast< SmoldynHub* >( e->data() )->
		reacConnectionFuncLocal( e, index, reac );
}

/**
 * This may as well come directly from a base class for hubs. It works
 * the same, with a different local getReacKf type function.
 */
void SmoldynHub::reacConnectionFuncLocal( 
		Element* hub, int rateTermIndex, Element* reac )
{
	// These fields will replace the original reaction fields so that
	// the lookups refer to the solver rather than the molecule.
	static Finfo* reacFields[] =
	{
		new ValueFinfo( "kf",
			ValueFtype1< double >::global(),
			GFCAST( &SmoldynHub::getReacKf ),
			RFCAST( &SmoldynHub::setReacKf )
		),
		new ValueFinfo( "kb",
			ValueFtype1< double >::global(),
			GFCAST( &SmoldynHub::getReacKb ),
			RFCAST( &SmoldynHub::setReacKb )
		),
	};
	static const ThisFinfo* tf = dynamic_cast< const ThisFinfo* >( 
		initReactionCinfo()->getThisFinfo( ) );
	assert( tf != 0 );
	static SolveFinfo reacZombieFinfo( 
		reacFields, 
		sizeof( reacFields ) / sizeof( Finfo* ),
		tf
	);

	zombify( hub, reac, reacSolveFinfo, &reacZombieFinfo );
	unsigned int connIndex = reacSolveFinfo->numOutgoing( hub );
	assert( connIndex > 0 ); // Should have just created a message on it
	assert( reacMap_.size() >= connIndex );

	reacMap_[connIndex - 1] = rateTermIndex;
}

void SmoldynHub::enzConnectionFunc( const Conn& c,
	unsigned int index, Element* enz )
{
	Element* e = c.targetElement();
	static_cast< SmoldynHub* >( e->data() )->
		enzConnectionFuncLocal( e, index, enz );
}

/**
 * Zombifies mmEnzs. But these pose an issue for Smoldyn: it is not a
 * good molecular concept. Two issues here. One is that the rate term
 * itself is odd, and we would have to munge the terms in Smoldyn to
 * be equivalent. the other is that the enz-substrate complex must not
 * deplete the originating enzymes.
 */
void SmoldynHub::mmEnzConnectionFunc( const Conn& c,
	unsigned int index, Element* mmEnz )
{
	// cout << "in mmEnzConnectionFunc for " << mmEnz->name() << endl;
	Element* e = c.targetElement();
	static_cast< SmoldynHub* >( e->data() )->
		mmEnzConnectionFuncLocal( e, index, mmEnz );
}

void SmoldynHub::enzConnectionFuncLocal(
	Element* hub, int rateTermIndex, Element* enz )
{
	static Finfo* enzFields[] =
	{
		new ValueFinfo( "k1",
			ValueFtype1< double >::global(),
			GFCAST( &SmoldynHub::getEnzK1 ),
			RFCAST( &SmoldynHub::setEnzK1 )
		),
		new ValueFinfo( "k2",
			ValueFtype1< double >::global(),
			GFCAST( &SmoldynHub::getEnzK2 ),
			RFCAST( &SmoldynHub::setEnzK2 )
		),
		new ValueFinfo( "k3",
			ValueFtype1< double >::global(),
			GFCAST( &SmoldynHub::getEnzK3 ),
			RFCAST( &SmoldynHub::setEnzK3 )
		),
		new ValueFinfo( "Km",
			ValueFtype1< double >::global(),
			GFCAST( &SmoldynHub::getEnzKm ),
			RFCAST( &SmoldynHub::setEnzKm )
		),
		new ValueFinfo( "kcat",
			ValueFtype1< double >::global(),
			GFCAST( &SmoldynHub::getEnzK3 ), // Same as k3
			RFCAST( &SmoldynHub::setEnzKcat ) // this is different.
		),
	};

	static const ThisFinfo* tf = dynamic_cast< const ThisFinfo* >( 
		initEnzymeCinfo()->getThisFinfo( ) );
	assert( tf != 0 );
	static SolveFinfo enzZombieFinfo( 
		enzFields, 
		sizeof( enzFields ) / sizeof( Finfo* ),
		tf
	);

	zombify( hub, enz, enzSolveFinfo, &enzZombieFinfo );
	unsigned int connIndex = enzSolveFinfo->numOutgoing( hub );
	assert( connIndex > 0 ); // Should have just created a message on it
	assert( enzMap_.size() >= connIndex );

	enzMap_[connIndex - 1] = rateTermIndex;
}

void SmoldynHub::mmEnzConnectionFuncLocal(
	Element* hub, int rateTermIndex, Element* mmEnz )
{
	static Finfo* enzFields[] =
	{
		new ValueFinfo( "k1",
			ValueFtype1< double >::global(),
			GFCAST( &SmoldynHub::getMmEnzK1 ),
			RFCAST( &SmoldynHub::setMmEnzK1 )
		),
		new ValueFinfo( "k2",
			ValueFtype1< double >::global(),
			GFCAST( &SmoldynHub::getMmEnzK2 ),
			RFCAST( &SmoldynHub::setMmEnzK2 )
		),
		new ValueFinfo( "k3",
			ValueFtype1< double >::global(),
			GFCAST( &SmoldynHub::getMmEnzKcat ),
			RFCAST( &SmoldynHub::setMmEnzK3 )
		),
		new ValueFinfo( "Km",
			ValueFtype1< double >::global(),
			GFCAST( &SmoldynHub::getMmEnzKm ),
			RFCAST( &SmoldynHub::setMmEnzKm )
		),
		new ValueFinfo( "kcat",
			ValueFtype1< double >::global(),
			GFCAST( &SmoldynHub::getMmEnzKcat ),
			RFCAST( &SmoldynHub::setMmEnzKcat )
		),
	};

	static const ThisFinfo* tf = dynamic_cast< const ThisFinfo* >( 
		initEnzymeCinfo()->getThisFinfo( ) );
	assert( tf != 0 );
	static SolveFinfo enzZombieFinfo( 
		enzFields, 
		sizeof( enzFields ) / sizeof( Finfo* ),
		tf
	);

	zombify( hub, mmEnz, mmEnzSolveFinfo, &enzZombieFinfo );
	unsigned int connIndex = mmEnzSolveFinfo->numOutgoing( hub );
	assert( connIndex > 0 ); // Should have just created a message on it
	assert( mmEnzMap_.size() >= connIndex );

	mmEnzMap_[connIndex - 1] = rateTermIndex;
}

/*
void unzombify( const Conn& c )
{
	Element* e = c.targetElement();
	const Cinfo* ci = e->cinfo();
	bool ret = ci->schedule( e );
	assert( ret );
	e->setThisFinfo( const_cast< Finfo* >( ci->getThisFinfo() ) );
	redirectDynamicMessages( e );
}
*/

/**
 * This operation turns the target element e into a zombie controlled
 * by the hub/solver. It gets rid of any process message coming into 
 * the zombie and replaces it with one from the solver.
 */
void SmoldynHub::zombify( 
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


/**
 * Clears out all the messages to zombie objects
 */
void SmoldynHub::clearFunc( const Conn& c )
{
	// cout << "Starting clearFunc for " << c.targetElement()->name() << endl;
	Element* e = c.targetElement();

	// First unzombify all targets
	vector< Conn > list;
	vector< Conn >::iterator i;

	molSolveFinfo->outgoingConns( e, list );
	// cout << "clearFunc: molSolveFinfo unzombified " << list.size() << " elements\n";
	molSolveFinfo->dropAll( e );
	for_each ( list.begin(), list.end(), unzombify );

	reacSolveFinfo->outgoingConns( e, list );
	// cout << "clearFunc: reacFinfo unzombified " << list.size() << " elements\n";
	reacSolveFinfo->dropAll( e );
	for_each ( list.begin(), list.end(), unzombify );

	enzSolveFinfo->outgoingConns( e, list );
	// cout << "clearFunc: enzFinfo unzombified " << list.size() << " elements\n";
	enzSolveFinfo->dropAll( e );
	for_each ( list.begin(), list.end(), unzombify );

	mmEnzSolveFinfo->outgoingConns( e, list );
	// cout << "clearFunc: mmEnzFinfo unzombified " << list.size() << " elements\n";
	mmEnzSolveFinfo->dropAll( e );
	for_each ( list.begin(), list.end(), unzombify );


	// Need the original molecule info. Where is that?
	// The molSumMap indexes from the sum index to the mol elist index.
	// But how do I access the mol elist? I need it to find the original
	// molecule element that was conneced from the table.
	molSumFinfo->incomingConns( e, list );
	molSumFinfo->dropAll( e );
}


///////////////////////////////////////////////////////////////////////
// Zombie functions. Later to be farmed out.
///////////////////////////////////////////////////////////////////////

/**
 * Here we provide the zombie function to set the 'n' field of the 
 * molecule. It first sets the solver location handling this
 * field, then the molecule itself.
 * For the molecule set/get operations, the lookup order is identical
 * to the message order. So we don't need an intermediate table.
 */
void SmoldynHub::setMolN( const Conn& c, double value )
{
}

double SmoldynHub::getMolN( const Element* e )
{
	return 0.0;
}

void SmoldynHub::setMolNinit( const Conn& c, double value )
{
}

double SmoldynHub::getMolNinit( const Element* e )
{
	return 0.0;
}

///////////////////////////////////////////////////
// Zombie object set/get function replacements for reactions
///////////////////////////////////////////////////

/**
 * Here we provide the zombie function to set the 'kf' field of the 
 * Reaction. It first sets the solver location handling this
 * field, then the reaction itself.
 * For the reaction set/get operations, the lookup order is
 * different from the message order. So we need an intermediate
 * table, the reacMap_, to map from one to the other.
 */
void SmoldynHub::setReacKf( const Conn& c, double value )
{
}

// getReacKf does not really need to go to the solver to get the value,
// because it should always remain in sync locally. But we do have
// to define the function to go with the set func in the replacement
// ValueFinfo.
double SmoldynHub::getReacKf( const Element* e )
{
	return 0.0;
}

void SmoldynHub::setReacKb( const Conn& c, double value )
{
}

double SmoldynHub::getReacKb( const Element* e )
{
	return 0.0;
}

///////////////////////////////////////////////////
// Zombie object set/get function replacements for Enzymes
///////////////////////////////////////////////////

/**
 * Here we provide the zombie function to set the 'k1' field of the 
 * Enzyme. It first sets the solver location handling this
 * field, then the reaction itself.
 * For the reaction set/get operations, the lookup order is
 * different from the message order. So we need an intermediate
 * table, the enzMap_, to map from one to the other.
 */
void SmoldynHub::setEnzK1( const Conn& c, double value )
{
}

// getEnzK1 does not really need to go to the solver to get the value,
// because it should always remain in sync locally. But we do have
// to define the function to go with the set func in the replacement
// ValueFinfo.
double SmoldynHub::getEnzK1( const Element* e )
{
	return 0.0;
}

void SmoldynHub::setEnzK2( const Conn& c, double value )
{
}

double SmoldynHub::getEnzK2( const Element* e )
{
	return 0.0;
}

void SmoldynHub::setEnzK3( const Conn& c, double value )
{
}

double SmoldynHub::getEnzK3( const Element* e )
{
	return 0.0;
}

// This function does rather nasty scaling of all rates so as to
// end up with the same overall Km when k3 is changed.
void SmoldynHub::setEnzKcat( const Conn& c, double value )
{
}


// This function does rather nasty scaling of all rates so as to
// end up with the same overall Km when k3 is changed.
void SmoldynHub::setEnzKm( const Conn& c, double value )
{
}

double SmoldynHub::getEnzKm( const Element* e )
{
	return 0.0;
}


//////////////////////////////////////////////////////////////////
// Here we set up stuff for mmEnzymes. It is similar, but not identical,
// to what we did for ordinary enzymes.
//////////////////////////////////////////////////////////////////
void SmoldynHub::setMmEnzK1( const Conn& c, double value )
{
}

// The Kinetic solver has no record of mmEnz::K1, so we simply go back to
// the object here. Should have a way to bypass this.
double SmoldynHub::getMmEnzK1( const Element* e )
{
	return 0.0;
}

// Ugh. Should perhaps ignore this mess.
void SmoldynHub::setMmEnzK2( const Conn& c, double value )
{
}

double SmoldynHub::getMmEnzK2( const Element* e )
{
	return 0.0;
}

// Note that this differs from assigning kcat. k3 leads to changes
// in Km and kcat, whereas kcat only affects kcat.
void SmoldynHub::setMmEnzK3( const Conn& c, double value )
{
}

double SmoldynHub::getMmEnzKcat( const Element* e )
{
	return 0.0;
}

void SmoldynHub::setMmEnzKcat( const Conn& c, double value )
{
}


// This function does rather nasty scaling of all rates so as to
// end up with the same overall Km when k3 is changed.
void SmoldynHub::setMmEnzKm( const Conn& c, double value )
{
}

double SmoldynHub::getMmEnzKm( const Element* e )
{
	return 0.0;
}

/////////////////////////////////////////////////////////////////////////
// Redirection functions
/////////////////////////////////////////////////////////////////////////

/**
 * This function redirects messages arriving at zombie elements onto
 * the hub. 
 * e is the zombie element whose messages are being redirected to the hub.
 * eFinfo is the Finfo holding those messages.
 * hubFinfo is the Finfo on the hub which will now handle the messages.
 * eIndex is the index to look up the element.
*/
/*
void redirectDestMessages(
	Element* hub, Element* e, const Finfo* hubFinfo, const Finfo* eFinfo,
	unsigned int eIndex, vector< unsigned int >& map, 
		vector< Element *>*  elist, bool retain )
{
	vector< Conn > clist;
	if ( eFinfo->incomingConns( e, clist ) == 0 )
		return;

	unsigned int i;
	unsigned int max = clist.size();
	vector< Element* > srcElements;
	vector< const Finfo* > srcFinfos;
	
	map.push_back( eIndex );
	
	// An issue here: Do I check if the src is on the solved tree?
	// This is a bad iteration: dropping connections is going to affect
	// the list size and position of i in the list.
	// for ( i = 0; i != max; i++ ) {
	i = max;
	while ( i > 0 ) {
		i--;
		Conn& c = clist[ i ];
		if ( find( elist->begin(), elist->end(), c.targetElement() ) == elist->end() )  {
			srcElements.push_back( c.targetElement() );
			srcFinfos.push_back( c.targetElement()->findFinfo( c.targetIndex() ) );
			if ( !retain )
				eFinfo->drop( e, i );
		}
	}
	// eFinfo->dropAll( e );
	for ( i = 0; i != srcElements.size(); i++ ) {
		bool ret = srcFinfos[ i ]->add( srcElements[ i ], hub, hubFinfo );
		assert( ret );
	}
}
*/

/**
 * Here we replace the existing DynamicFinfos and their messages with
 * new ones for the updated access functions
 */
/*
void redirectDynamicMessages( Element* e )
{
	const Finfo* f;
	unsigned int finfoNum = 1;
	unsigned int i;

	vector< Conn > clist;

	while ( ( f = e->localFinfo( finfoNum ) ) ) {
		const DynamicFinfo *df = dynamic_cast< const DynamicFinfo* >( f );
		assert( df != 0 );
		f->incomingConns( e, clist );
		unsigned int max = clist.size();
		vector< Element* > srcElements( max );
		vector< const Finfo* > srcFinfos( max );
		// An issue here: Do I check if the src is on the solved tree?
		for ( i = 0; i != max; i++ ) {
			Conn& c = clist[ i ];
			srcElements[ i ] = c.targetElement();
			srcFinfos[ i ]= c.targetElement()->findFinfo( c.targetIndex() );
		}

		f->outgoingConns( e, clist );
		max = clist.size();
		vector< Element* > destElements( max );
		vector< const Finfo* > destFinfos( max );
		for ( i = 0; i != max; i++ ) {
			Conn& c = clist[ i ];
			destElements[ i ] = c.targetElement();
			destFinfos[ i ] = c.targetElement()->findFinfo( c.targetIndex() );
		}
		string name = df->name();
		bool ret = e->dropFinfo( df );
		assert( ret );

		const Finfo* origFinfo = e->findFinfo( name );
		assert( origFinfo );

		max = srcFinfos.size();
		for ( i =  0; i < max; i++ ) {
			ret = srcFinfos[ i ]->add( srcElements[ i ], e, origFinfo );
			assert( ret );
		}
		max = destFinfos.size();
		for ( i =  0; i < max; i++ ) {
			ret = origFinfo->add( e, destElements[ i ], destFinfos[ i ] );
			assert( ret );
		}

		finfoNum++;
	}
}
*/
