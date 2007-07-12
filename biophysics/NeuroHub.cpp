/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "moose.h"
#include "HSolveStructure.h"
#include "NeuroHub.h"
#include "Compartment.h"
#include "ThisFinfo.h"
#include "SolveFinfo.h"

const Cinfo* initNeuroHubCinfo()
{
	/**
	 * This is identical to the message sent from clock Ticks to
	 * objects. Here it is used to take over the Process message,
	 * usually only as a handle from the solver to the object.
	 */
	static Finfo* zombieShared[] =
	{
		new SrcFinfo( "process", Ftype1< ProcInfo >::global() ),
		new SrcFinfo( "reinit", Ftype1< ProcInfo >::global() ),
	};

	/**
	 * This is the destination of the several messages from the scanner.
	 */
	static Finfo* hubShared[] =
	{
		new DestFinfo( "compartment",
			Ftype1< vector< Element* >* >::global(),
			RFCAST( &NeuroHub::compartmentFunc ) ),
		new DestFinfo( "channel",
			Ftype1< vector< Element* >* >::global(),
			RFCAST( &NeuroHub::channelFunc ) ),
		new DestFinfo( "gate",
			Ftype1< vector< Element* >* >::global(),
			RFCAST( &NeuroHub::gateFunc ) ),
	};

	static Finfo* neuroHubFinfos[] =
	{
	///////////////////////////////////////////////////////
	// Field definitions
	///////////////////////////////////////////////////////
	///////////////////////////////////////////////////////
	// MsgSrc definitions
	///////////////////////////////////////////////////////
	///////////////////////////////////////////////////////
	// MsgDest definitions
	///////////////////////////////////////////////////////
		new DestFinfo( "destroy", Ftype0::global(),
			&NeuroHub::destroy ),
	///////////////////////////////////////////////////////
	// Shared definitions
	///////////////////////////////////////////////////////
		new SharedFinfo( "hub", hubShared, 
			sizeof( hubShared ) / sizeof( Finfo* ) ),
		new SharedFinfo( "comptSolve", zombieShared, 
			sizeof( zombieShared ) / sizeof( Finfo* ) ),
	};

	static Cinfo neuroHubCinfo(
		"NeuroHub",
		"Niraj Dudani, 2007, NCBS",
		"NeuroHub: Object for controlling reaction systems on behalf of the\nStoich object. Interfaces both with the reaction system\n(molecules, reactions, enzymes\nand user defined rate terms) and also with the Stoich\nclass which generates the stoichiometry matrix and \nhandles the derivative calculations.",
		initNeutralCinfo(),
		neuroHubFinfos,
		sizeof( neuroHubFinfos ) / sizeof( Finfo* ),
		ValueFtype1< NeuroHub >::global()
	);

	return &neuroHubCinfo;
}

static const Cinfo* neuroHubCinfo = initNeuroHubCinfo();

static const Finfo* comptSolveFinfo = 
	initNeuroHubCinfo()->findFinfo( "comptSolve" );

///////////////////////////////////////////////////
// Field access functions (for Hub)
///////////////////////////////////////////////////

///////////////////////////////////////////////////
// Dest functions (for Hub)
///////////////////////////////////////////////////

void NeuroHub::compartmentFunc( const Conn& c,
	vector< Element* >* elist )
{
	Element* hub = c.targetElement();
	static_cast< NeuroHub* >( hub->data() )->
		innerCompartmentFunc( hub, elist );
}

void NeuroHub::innerCompartmentFunc( Element* hub, vector< Element* >* elist )
{
	// These fields will replace the original compartment fields so that
	// the lookups refer to the solver rather than the compartment.
	static Finfo* comptFields[] =
	{
		new ValueFinfo( "Vm",
			ValueFtype1< double >::global(),
			GFCAST( &NeuroHub::getComptVm ),
			RFCAST( &NeuroHub::setComptVm )
		),
		new ValueFinfo( "inject",
			ValueFtype1< double >::global(),
			GFCAST( &NeuroHub::getInject ),
			RFCAST( &NeuroHub::setInject )
		),
	};
	
	static const ThisFinfo* tf = dynamic_cast< const ThisFinfo* >( 
		initCompartmentCinfo()->getThisFinfo( ) );
	assert( tf != 0 );
	static SolveFinfo comptZombieFinfo( 
		comptFields, 
		sizeof( comptFields ) / sizeof( Finfo* ),
		tf
	);
	
	vector< Element* >::iterator i;
	for ( i = elist->begin(); i != elist->end(); ++i ) {
		zombify( hub, *i,
			 comptSolveFinfo, &comptZombieFinfo );
//		redirectDestMessages( hub, *i, molSumFinfo, sumTotFinfo );
		redirectDynamicMessages( *i );
	}
}

void NeuroHub::channelFunc( const Conn& c,
	vector< Element* >* elist )
{
	;
}

void NeuroHub::gateFunc( const Conn& c,
	vector< Element* >* elist )
{
	;
}

/**
 * Overrides Neutral::destroy to clean up zombies.
 * In this destructor we need to put messages back to process,
 * and we need to replace the SolveFinfos on zombies with the
 * original ThisFinfo.
 */
void NeuroHub::destroy( const Conn& c)
{
  /*	static Finfo* origMolFinfo =
		const_cast< Finfo* >(
		initMoleculeCinfo()->getThisFinfo( ) );
	static Finfo* origReacFinfo =
		const_cast< Finfo* >(
		initReactionCinfo()->getThisFinfo( ) );
	Element* hub = c.targetElement();
	vector< Conn > targets;
	vector< Conn >::iterator i;

	// First (todo) put the messages back onto the scheduler.
	// Second, replace the SolveFinfos
	molSolveFinfo->outgoingConns( hub, targets );
	for ( i = targets.begin(); i != targets.end(); i++ )
		i->targetElement()->setThisFinfo( origMolFinfo );

	reacSolveFinfo->outgoingConns( hub, targets );
	for ( i = targets.begin(); i != targets.end(); i++ )
		i->targetElement()->setThisFinfo( origReacFinfo );

	Neutral::destroy( c );
  */
}

///////////////////////////////////////////////////
// Field access functions (Biophysics)
///////////////////////////////////////////////////

/**
 * Here we provide the zombie function to set the 'Vm' field of the 
 * compartment. It first sets the solver location handling this
 * field, then the compartment itself.
 * For the compartment set/get operations, the lookup order is identical
 * to the message order. So we don't need an intermediate table.
 */
void NeuroHub::setComptVm( const Conn& c, double value )
{
	unsigned int comptIndex;
	NeuroHub* nh = getHubFromZombie( 
		c.targetElement(), comptSolveFinfo, comptIndex );
	if ( nh ) {
		assert ( comptIndex < nh->V_.size() );
		( nh->V_ )[ comptIndex ] = value;
	}
	// Required?
	Compartment::setVm( c, value );
}

double NeuroHub::getComptVm( const Element* e )
{
	unsigned int comptIndex;
	NeuroHub* nh = getHubFromZombie( e, comptSolveFinfo, comptIndex );
	if ( nh ) {
		assert ( comptIndex < nh->V_.size() );
		return ( nh->V_ )[ comptIndex ];
	}
	return 0.0;
}

void NeuroHub::setInject( const Conn& c, double value )
{
	unsigned int comptIndex;
	NeuroHub* nh = getHubFromZombie( 
		c.targetElement(), comptSolveFinfo, comptIndex );
	if ( nh ) {
		assert ( comptIndex < nh->inject_.size() );
		( nh->inject_ )[ comptIndex ] = value;
	}
	// Required?
	Compartment::setInject( c, value );
}

double NeuroHub::getInject( const Element* e )
{
	unsigned int comptIndex;
	NeuroHub* nh = getHubFromZombie( e, comptSolveFinfo, comptIndex );
	if ( nh ) {
		assert ( comptIndex < nh->inject_.size() );
		return ( nh->inject_ )[ comptIndex ];
	}
	return 0.0;
}

///////////////////////////////////////////////////
// Dest functions (Biophysics)
///////////////////////////////////////////////////
void NeuroHub::comptInjectMsgFunc( const Conn& c, double I )
{
/*
	Compartment* compt = static_cast< Compartment* >(
		c.targetElement()->data() );
	compt->sumInject_ += I;
	compt->Im_ += I;
*/
}

///////////////////////////////////////////////////
// Utility functions
///////////////////////////////////////////////////
/**
 * This operation turns the target element e into a zombie controlled
 * by the hub/solver. It gets rid of any process message coming into 
 * the zombie and replaces it with one from the solver.
 */
void NeuroHub::zombify(
	Element* hub, Element* e,
	const Finfo* hubFinfo, Finfo* solveFinfo )
{
	// Replace the original procFinfo with one from the hub.
	const Finfo* procFinfo = e->findFinfo( "process" );
	procFinfo->dropAll( e );
	bool ret =
		hubFinfo->add( hub, e, procFinfo );
	assert( ret );
	
	// Redirect original messages from the zombie to the hub.
	// Pending.
	
	// Replace the 'ThisFinfo' on the solved element
	e->setThisFinfo( solveFinfo );
}

/**
 * This function redirects messages arriving at zombie elements onto
 * the hub. 
 * e is the zombie element whose messages are being redirected to the hub.
 * eFinfo is the Finfo holding those messages.
 * hubFinfo is the Finfo on the hub which will now handle the messages.
 * eIndex is the index to look up the element.
 */
void NeuroHub::redirectDestMessages(
	Element* hub, Element* e,
	const Finfo* hubFinfo, const Finfo* eFinfo,
	unsigned int eIndex = 0, vector< unsigned int >* map = 0 )
{
	vector< Conn > clist;
	if ( eFinfo->incomingConns( e, clist ) == 0 )
		return;
	
	unsigned int i;
	unsigned int max = clist.size();
	vector< Element* > srcElements( max );
	vector< const Finfo* > srcFinfos( max );
	
	if ( map )
		map->push_back( eIndex );
	
	// An issue here: Do I check if the src is on the solved tree?
	for ( i = 0; i != max; i++ ) {
		Conn& c = clist[ i ];
		srcElements[ i ] = c.targetElement();
		srcFinfos[ i ]   = c.targetElement()->
			findFinfo( c.targetIndex() );
	}
	eFinfo->dropAll( e );
	for ( i = 0; i != max; i++ ) {
		srcFinfos[ i ]->add( srcElements[ i ], hub, hubFinfo );
	}
}

/**
 * Here we replace the existing DynamicFinfos and their messages with
 * new ones for the updated access functions
 */
void NeuroHub::redirectDynamicMessages( Element* e )
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

/**
 * Looks up the solver from the zombie element e. Returns the solver
 * element, or null on failure. 
 * It needs the originating Finfo on the solver that connects to the zombie,
 * as the srcFinfo.
 * Also passes back the index of the zombie element on this set of
 * messages. This is NOT the absolute Conn index.
 */
NeuroHub* NeuroHub::getHubFromZombie(
	const Element* e, const Finfo* srcFinfo,
	unsigned int& index )
{
	const SolveFinfo* f = dynamic_cast< const SolveFinfo* > (
		e->getThisFinfo() );
	if ( !f ) return 0;
	const Conn& c = f->getSolvedConn( e );
	unsigned int slot;
	srcFinfo->getSlotIndex( srcFinfo->name(), slot );
	Element* hub = c.targetElement();
	index = hub->connSrcRelativeIndex( c, slot );
	return static_cast< NeuroHub* >( hub->data() );
}
