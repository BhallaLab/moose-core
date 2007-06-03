/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/
#include "moose.h"
#include "Tick.h"
#include "ParTick.h"

/**
 * The ParTick handles scheduling in a parallel simulation. It works
 * closely with the PostMaster and does reordering of the
 * execution of operations so as to interleave computation and 
 * communication. Like the Tick, it sends the
 * Process (or other event) message to all the scheduled objects,
 * and it keeps track of the update sequence with its sibling ParTick
 * objects. In addition, it divides each tick into 5 stages:
 *
 * Stage 0: Post irecv. This is non-blocking and sets up storage to
 * 		handle incoming data for this tick.
 * Stage 1: Call all processes that have to send data out on this tick
 * Stage 2: All the outgoing data has arrived at Postmaster. Post send.
 * 		Note that this is a blocking call, but it will typically only
 * 		wait if the target irecv has not yet been posted.
 * Stage 3: Call all processes that work only with local data.
 * Stage 4: Poll for posted irecvs. As they arrive send their contents
 * 		to the destination objects.
 * 		The poll process relies on return info from each PostMaster.
 *
 * Stage 0, 2, 4 deal with the postmaster.
 * Stage 1, 3 deal with regular objects.
 * Over the course of this cycle, all information that was generated
 * on this clock tick will have reached its target, regardless of which
 * node it is on. 
 *
 */

const Cinfo* initParTickCinfo()
{
	/**
	 * Here we inherit most of the tick-sequencing code from Tick.
	 * This works because we use the virtual functions 
	 * innerProcessFunc and innerReinitFunc calls to deal with the
	 * Process and Reinit calls.
	 *
	 * The inherited fields are
	 *	SharedFinfo "next"
	 *	SharedFinfo "prev"
	 *	SharedFinfo "process"
	 *		Note that here the 'process' message is used for objects
	 *		that have only local messaging. We need to define a new
	 *		call, below for objects that have outgoing messages.
	 *		Also note that the type info is identical, reuse the 
	 *		old processTypes.
	 *	ValueFinfo: dt
	 *	ValueFinfo: stage
	 *	ValueFinfo: ordinal
	 *	ValueFinfo: nextTime
	 *	ValueFinfo: path
	 *	DestFinfo "newObject"
	 *	DestFinfo "start"
	 *	SrcFinfo "processSrc"
	 *	SrcFinfo "updateDtSrc"
	 */

	/**
	 * This goes to all scheduled objects to call their process events.
	 * Although it is identical to the one in Tick.cpp, we redo it
	 * because of scope issues.
	 */
	static TypeFuncPair processTypes[] = 
	{
		// The process function call
		TypeFuncPair( Ftype1< ProcInfo >::global(), 0 ),
		// The reinit function call
		TypeFuncPair( Ftype1< ProcInfo >::global(), 0 ),
	};

	/**
	 * This shared message communicates with the postmaster
	 */
	static TypeFuncPair parTypes[] = 
	{
		// This first entry is to tell the PostMaster to post iRecvs
		// The argument is the ordinal number of the clock tick
		TypeFuncPair( Ftype1< int >::global(), 0 ),
		// The second entry is to tell the PostMaster to post 'send'
		TypeFuncPair( Ftype1< int >::global(), 0 ),
		// The third entry is for polling the receipt of incoming data.
		// Each PostMaster does an MPI_Test on the earlier posted iRecv.
		TypeFuncPair( Ftype1< int >::global(), 0 ),
		// The fourth entry is for harvesting the poll request.
		// The argument is the node number handled by the postmaster.
		// It comes back when the polling on that postmaster is done.
		TypeFuncPair( Ftype1< unsigned int >::global(),
						RFCAST( &ParTick::pollFunc ) )
	};


	static Finfo* parTickFinfos[] =
	{
	///////////////////////////////////////////////////////
	// Field definitions
	///////////////////////////////////////////////////////
			/*
		new ValueFinfo( "dt", ValueFtype1< double >::global(),
			GFCAST( &Tick::getDt ),
			RFCAST( &Tick::setDt )
		),
		*/
	///////////////////////////////////////////////////////
	// Shared message definitions
	///////////////////////////////////////////////////////
		new SharedFinfo( "outgoingProcess", processTypes, 2 ),
		new SharedFinfo( "parTick", parTypes, 
			sizeof( parTypes ) / sizeof( TypeFuncPair ) ),
	
	///////////////////////////////////////////////////////
	// MsgSrc definitions
	///////////////////////////////////////////////////////
		// new SrcFinfo( "processSrc", Ftype1< ProcInfo >::global() ),

	///////////////////////////////////////////////////////
	// MsgDest definitions
	///////////////////////////////////////////////////////

		/**
		 * The poll function keeps track of how many postmasters
		 * have finished their irecv.
		 * Actually this is all part of the shared message to the
		 * postmaster.
		new DestFinfo( "poll",
			Ftype0::global(),
			RFCAST( &ParTick::poll ) ),
		 */
	};
	
	static Cinfo parTickCinfo(
		"ParTick",
		"Upinder S. Bhalla, April 2007, NCBS",
		"ParTick: Sequences execution of objects on a given dt for working with parallel messaging. Interleaves computation and communication for efficiency.",
		initTickCinfo(),
		parTickFinfos,
		sizeof(parTickFinfos)/sizeof(Finfo *),
		ValueFtype1< ParTick >::global()
	);

	return &parTickCinfo;
}

static const Cinfo* parTickCinfo = initParTickCinfo();

static const unsigned int processSlot = 
	initParTickCinfo()->getSlotIndex( "process" ) + 0;
static const unsigned int reinitSlot = 
	initParTickCinfo()->getSlotIndex( "process" ) + 1;

static const unsigned int outgoingProcessSlot = 
	initParTickCinfo()->getSlotIndex( "outgoingProcess" ) + 0;
static const unsigned int outgoingReinitSlot = 
	initParTickCinfo()->getSlotIndex( "outgoingProcess" ) + 1;

static const unsigned int iRecvSlot = 
	initParTickCinfo()->getSlotIndex( "parTick" ) + 0;
static const unsigned int sendSlot = 
	initParTickCinfo()->getSlotIndex( "parTick" ) + 1;
static const unsigned int pollSlot = 
	initParTickCinfo()->getSlotIndex( "parTick" ) + 2;

///////////////////////////////////////////////////
// Field function definitions
///////////////////////////////////////////////////


///////////////////////////////////////////////////
// Dest function definitions
///////////////////////////////////////////////////

void ParTick::pollFunc( const Conn& c, unsigned int node ) 
{
	static_cast< ParTick* >( c.data() )->innerPollFunc( node );
}

void ParTick::innerPollFunc( unsigned int node ) 
{
	assert( node < pendingNodes_.size() );
	assert( pendingNodes_[node] == 1 );
	pendingNodes_[node] = 0;
	--pendingCount_;
}

bool ParTick::pendingData() const
{
	return ( pendingCount_ != 0 );
}

/**
 * This is a virtual function that handles the issuing of 
 * Process calls to all targets. For the ParTick, this means
 * sequencing the outgoing calls into 5 stages.
 * Stage 0: Post irecv. This is non-blocking and sets up storage to
 * 		handle incoming data for this tick.
 * Stage 1: Call all processes that have to send data out on this tick
 * Stage 2: All the outgoing data has arrived at Postmaster. Post send.
 * 		Note that this is a blocking call, but it will typically only
 * 		wait if the target irecv has not yet been posted.
 * Stage 3: Call all processes that work only with local data.
 * Stage 4: Poll for posted irecvs. As they arrive send their contents
 * 		to the destination objects.
 * 		The poll process relies on return info from each PostMaster.
 */
void ParTick::innerProcessFunc( Element* e, ProcInfo info )
{
	// Phase 0: post iRecv
	send1< int >( e, iRecvSlot, ordinal() );
	// Phase 1: call Process for objects connected off-node
	send1< ProcInfo >( e, outgoingProcessSlot, info );
	// Phase 2: send data off node
	send1< int >( e, sendSlot, ordinal() );
	// Phase 3: Call regular process for locally connected objects
	send1< ProcInfo >( e, processSlot, info );
	// Phase 4: Poll for arrival of all data
	initPending( e );
	while( pendingData() ) {
		// cout << "." << flush;
		send1< int >( e, pollSlot, ordinal() );
	}
}

void ParTick::initPending( Element* e )
{
	static const Finfo* parFinfo = e->findFinfo( "parTick" );
	pendingCount_ = parFinfo->numOutgoing( e );
	cout << "pendingCount = " << pendingCount_ << endl;
	pendingNodes_.resize( pendingCount_ + 1 );
	pendingNodes_.assign( pendingCount_ + 1, 1);
}

/**
 * Similar virtual function to deal with managing reinit info going out
 * to other nodes.
 */ 
void ParTick::innerReinitFunc( Element* e, ProcInfo info )
{
	// Here we need to scan all managed objects and decide if they
	// need to be on the outgoing process list or if they are local.
	//
	// separateOutgoingTargets
	//

	// Phase 0: post iRecv
	send1< int >( e, iRecvSlot, ordinal() );
	// Phase 1: call Reinit for objects connected off-node
	send1< ProcInfo >( e, outgoingReinitSlot, info );
	// Phase 2: send data off node
	send1< int >( e, sendSlot, ordinal() );
	// Phase 3: Call regular reinit for locally connected objects
	send1< ProcInfo >( e, reinitSlot, info );
	// Phase 4: Poll for arrival of all data
	initPending( e );
	while( pendingData() )
		send1< int >( e, pollSlot, ordinal() );
}


///////////////////////////////////////////////////
// Other function definitions
///////////////////////////////////////////////////
