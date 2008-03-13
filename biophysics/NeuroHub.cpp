/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**   copyright (C) 2003-2007 Upinder S. Bhalla, Niraj Dudani and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "moose.h"
#include "../element/Neutral.h"
#include <queue>
#include "SynInfo.h"
#include "HSolveStruct.h"
#include "NeuroHub.h"
#include "Compartment.h"
#include "HHChannel.h"
#include "SpikeGen.h"
#include "SynChan.h"
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
		new DestFinfo( "spikegen",
			Ftype1< vector< Element* >* >::global(),
			RFCAST( &NeuroHub::spikegenFunc ) ),
		new DestFinfo( "synchan",
			Ftype1< vector< Element* >* >::global(),
			RFCAST( &NeuroHub::synchanFunc ) ),
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
		// override the Neutral::childFunc here, so that when this
		// is deleted all the zombies are reanimated.
		new DestFinfo( "child", Ftype1< int >::global(),
			RFCAST( &NeuroHub::childFunc ) ),
	///////////////////////////////////////////////////////
	// Shared definitions
	///////////////////////////////////////////////////////
		new SharedFinfo( "hub", hubShared, 
			sizeof( hubShared ) / sizeof( Finfo* ) ),
		new SharedFinfo( "comptSolve", zombieShared, 
			sizeof( zombieShared ) / sizeof( Finfo* ) ),
		new SharedFinfo( "chanSolve", zombieShared, 
			sizeof( zombieShared ) / sizeof( Finfo* ) ),
		new SharedFinfo( "spikeSolve", zombieShared, 
			sizeof( zombieShared ) / sizeof( Finfo* ) ),
		new SharedFinfo( "synSolve", zombieShared, 
			sizeof( zombieShared ) / sizeof( Finfo* ) ),
	};
	
	static Cinfo neuroHubCinfo(
		"NeuroHub",
		"Niraj Dudani, 2007, NCBS",
		"NeuroHub: ",
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
static const Finfo* chanSolveFinfo = 
	initNeuroHubCinfo()->findFinfo( "chanSolve" );
static const Finfo* spikeSolveFinfo = 
	initNeuroHubCinfo()->findFinfo( "spikeSolve" );
static const Finfo* synSolveFinfo = 
	initNeuroHubCinfo()->findFinfo( "synSolve" );

///////////////////////////////////////////////////
// Field access functions (for Hub)
///////////////////////////////////////////////////

///////////////////////////////////////////////////
// Dest functions (for Hub)
///////////////////////////////////////////////////

void NeuroHub::compartmentFunc( const Conn* c,
	vector< Element* >* elist )
{
	Element* hub = c->targetElement();
	static_cast< NeuroHub* >( c->data() )->innerCompartmentFunc( hub, elist );
}

void NeuroHub::channelFunc( const Conn* c,
	vector< Element* >* elist )
{
	Element* hub = c->targetElement();
	static_cast< NeuroHub* >( c->data() )->innerChannelFunc( hub, elist );
}

void NeuroHub::spikegenFunc( const Conn* c,
	vector< Element* >* elist )
{
	Element* hub = c->targetElement();
	static_cast< NeuroHub* >( c->data() )->innerSpikegenFunc( hub, elist );
}

void NeuroHub::synchanFunc( const Conn* c,
	vector< Element* >* elist )
{
	Element* hub = c->targetElement();
	static_cast< NeuroHub* >( c->data() )->innerSynchanFunc( hub, elist );
}

void NeuroHub::innerCompartmentFunc(
	Element* hub,
	vector< Element* >* elist )
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
			 comptSolveFinfo,
			 &comptZombieFinfo );
		// Compartment receives 2 messages from Tick process
		const Finfo* initFinfo = ( *i )->findFinfo( "init" );
		initFinfo->dropAll( *i );
//		redirectDestMessages( hub, *i, molSumFinfo, sumTotFinfo );
		redirectDynamicMessages( *i );
	}
}

void NeuroHub::innerChannelFunc(
	Element* hub,
	vector< Element* >* elist )
{
	static Finfo* chanFields[] =
	{
		new ValueFinfo( "Gbar", ValueFtype1< double >::global(),
			GFCAST( &NeuroHub::getChanGbar ), 
			RFCAST( &NeuroHub::setChanGbar )
		),
	};
	
	static const ThisFinfo* tf = dynamic_cast< const ThisFinfo* >( 
		initHHChannelCinfo()->getThisFinfo( ) );
	assert( tf != 0 );
	static SolveFinfo chanZombieFinfo( 
		chanFields, 
		sizeof( chanFields ) / sizeof( Finfo* ),
		tf
	);
	
	vector< Element* >::iterator i;
	for ( i = elist->begin(); i != elist->end(); ++i ) {
		zombify( hub, *i,
			 chanSolveFinfo,
			 &chanZombieFinfo );
//		redirectDestMessages( hub, *i, molSumFinfo, sumTotFinfo );
		redirectDynamicMessages( *i );
	}
}

void NeuroHub::innerSpikegenFunc(
	Element* hub,
	vector< Element* >* elist )
{
	vector< Element* >::iterator i;
	for ( i = elist->begin(); i != elist->end(); ++i ) {
		const Finfo* procFinfo = ( *i )->findFinfo( "process" );
		procFinfo->dropAll( *i );
		bool ret = spikeSolveFinfo->add( hub, *i, procFinfo );
		assert( ret );
//		redirectDestMessages( hub, *i, molSumFinfo, sumTotFinfo );
		redirectDynamicMessages( *i );
	}
}

void NeuroHub::innerSynchanFunc(
	Element* hub,
	vector< Element* >* elist )
{
	static Finfo* synFields[] =
	{
		new ValueFinfo( "Gbar", ValueFtype1< double >::global(),
			GFCAST( &NeuroHub::getSynChanGbar ), 
			RFCAST( &NeuroHub::setSynChanGbar )
		),
	};

	static const ThisFinfo* tf = dynamic_cast< const ThisFinfo* >( 
		initSynChanCinfo()->getThisFinfo( ) );
	assert( tf != 0 );
	static SolveFinfo synZombieFinfo( 
		synFields, 
		sizeof( synFields ) / sizeof( Finfo* ),
		tf
	);
	
	vector< Element* >::iterator i;
	for ( i = elist->begin(); i != elist->end(); ++i ) {
		zombify( hub, *i,
			 synSolveFinfo,
			 &synZombieFinfo );
//		redirectDestMessages( hub, *i, molSumFinfo, sumTotFinfo );
		redirectDynamicMessages( *i );
	}
}

/**
 * Overrides Neutral::destroy to clean up zombies.
 * In this destructor we need to put messages back to process,
 * and we need to replace the SolveFinfos on zombies with the
 * original ThisFinfo.
 */
void NeuroHub::destroy( const Conn* c )
{
/*
	static Finfo* origMolFinfo =
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
*/
	Neutral::destroy( c );
}

void NeuroHub::childFunc( const Conn* c, int stage )
{
	if ( stage == 1 ) // clear messages: first clean out zombies before
		// the messages are all deleted.
		clearFunc( c );
	// Then fall back into what the Neutral version does
	Neutral::childFunc( c, stage );
}

void NeuroHub::unzombify( const Conn* c )
{
	Element* e = c->targetElement();
	const Cinfo* ci = e->cinfo();
	bool ret = ci->schedule( e );
	assert( ret );
	e->setThisFinfo( const_cast< Finfo* >( ci->getThisFinfo() ) );
//	redirectDynamicMessages( e );
}

/**
 * Clears out all the messages to zombie objects
 */
void NeuroHub::clearFunc( const Conn* c )
{
	Element* e = c->targetElement();
	
	// First unzombify all targets
	vector< Conn > list;
	vector< Conn >::iterator i;
	
	comptSolveFinfo->outgoingConns( e, list );
	comptSolveFinfo->dropAll( e );
	// I'll soon enough reinstate all this.
	// for_each ( list.begin(), list.end(), unzombify );
	for( i = list.begin(); i != list.end(); i++ ) unzombify( &( *i ) );
	
	chanSolveFinfo->outgoingConns( e, list );
	chanSolveFinfo->dropAll( e );
	// for_each ( list.begin(), list.end(), unzombify );
	for( i = list.begin(); i != list.end(); i++ ) unzombify( &( *i ) );
	
	spikeSolveFinfo->outgoingConns( e, list );
	spikeSolveFinfo->dropAll( e );
	// for_each ( list.begin(), list.end(), unzombify );
	for( i = list.begin(); i != list.end(); i++ ) unzombify( &( *i ) );
	
	synSolveFinfo->outgoingConns( e, list );
	synSolveFinfo->dropAll( e );
	// for_each ( list.begin(), list.end(), unzombify );
	for( i = list.begin(); i != list.end(); i++ ) unzombify( &( *i ) );
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
void NeuroHub::setComptVm( const Conn* c, double value )
{
	unsigned int comptIndex;
	NeuroHub* nh = getHubFromZombie( 
		c->targetElement(), comptSolveFinfo, comptIndex );
	if ( nh ) {
		assert ( comptIndex < nh->V_.size() );
		( nh->V_ )[ comptIndex ] = value;
	}
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

void NeuroHub::setInject( const Conn* c, double value )
{
	unsigned int comptIndex;
	NeuroHub* nh = getHubFromZombie( 
		c->targetElement(), comptSolveFinfo, comptIndex );
	if ( nh ) {
		assert ( comptIndex < nh->inject_.size() );
		( nh->inject_ )[ comptIndex ] = value;
	}
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

void NeuroHub::setChanGbar( const Conn* c, double value )
{
	;
}

double NeuroHub::getChanGbar( const Element* e )
{
	return 0.0;
}

void NeuroHub::setSynChanGbar( const Conn* c, double value )
{
	;
}

double NeuroHub::getSynChanGbar( const Element* e )
{
	return 0.0;
}

///////////////////////////////////////////////////
// Dest functions (Biophysics)
///////////////////////////////////////////////////
void NeuroHub::comptInjectMsgFunc( const Conn* c, double I )
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
	bool ret = hubFinfo->add( hub, e, procFinfo );
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
	const Conn* c = f->getSolvedConn( e );
	Slot slot;
	srcFinfo->getSlot( srcFinfo->name(), slot );
	Element* hub = c->targetElement();
	index = hub->connSrcRelativeIndex( c, slot.msg() );
	return static_cast< NeuroHub* >( hub->data() );
}
