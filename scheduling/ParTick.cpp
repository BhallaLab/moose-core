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
#include "../shell/Shell.h"

// #define PRINT_POS

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
	static Finfo* processShared[] =
	{
		new SrcFinfo( "process", Ftype1< ProcInfo >::global(),
			"The process function call" ),
		new SrcFinfo( "reinit", Ftype1< ProcInfo >::global(),
			"The reinit function call" ),
	};
	static Finfo* parShared[] =
	{
		new SrcFinfo( "postIrecv", Ftype0::global(),
			"This first entry is to tell the PostMaster to post iRecvs" ),
		new SrcFinfo( "postSend",  Ftype1< bool >::global(),
			"The second entry is to tell the PostMaster to post 'send'" ),
		new SrcFinfo( "poll", Ftype1< bool >::global(), 
			"The third entry is for polling the receipt of incoming data.\n"
			"Each PostMaster does an MPI_Test on the earlier posted iRecv." ),
		new DestFinfo( "harvestPoll", Ftype1< unsigned int >::global(),
						RFCAST( &ParTick::pollFunc ),
						"The fourth entry is for harvesting the poll request.The argument "
						"is the node number handled by the postmaster.It comes back when the "
						"polling on that postmaster is done." ),
		new SrcFinfo( "clearSetupStack", Ftype0::global(),
			"This entry tells the postMaster to execute pending setup ops.These are blocking calls, "
			"but they may invoke nested polling calls." ),
		new SrcFinfo( "barrier", Ftype0::global(),
			"The last entry is to tell targets to execute a Barrier command, used to synchronize all "
			"nodes. Warning: this message should only be called on a single target postmaster using "
			"sendTo.Otherwise each target postmaster will try to set a barrier." ),
	};


	static Finfo* parTickFinfos[] =
	{
	///////////////////////////////////////////////////////
	// Field definitions
	///////////////////////////////////////////////////////
		new ValueFinfo( "barrier", ValueFtype1< int >::global(),
			GFCAST( &ParTick::getBarrier ),
			RFCAST( &ParTick::setBarrier )
		),
		new ValueFinfo( "doSync", ValueFtype1< bool >::global(),
			GFCAST( &ParTick::getSync ),
			RFCAST( &ParTick::setSync )
		),
	///////////////////////////////////////////////////////
	// Shared message definitions
	///////////////////////////////////////////////////////
		new SharedFinfo( "outgoingProcess", processShared,
			sizeof( processShared ) / sizeof( Finfo* ),
			"This goes to all scheduled objects to call their process events.Although it is identical "
			"to the one in Tick.cpp, we redo it  because of scope issues." ),
		new SharedFinfo( "parTick", parShared, 
			sizeof( parShared ) / sizeof( Finfo* ),
			"This shared message communicates with the postmaster" ),
	
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
	
	static string doc[] =
	{
		"Name", "ParTick",
		"Author", "Upinder S. Bhalla, April 2007, NCBS",
		"Description", "ParTick: Sequences execution of objects on a given dt for working with "
				"parallel messaging. Interleaves computation and communication for efficiency.",
	};
	static Cinfo parTickCinfo(
		doc,
		sizeof( doc ) / sizeof( string ),
		initTickCinfo(),
		parTickFinfos,
		sizeof(parTickFinfos)/sizeof(Finfo *),
		ValueFtype1< ParTick >::global()
	);

	return &parTickCinfo;
}

static const Cinfo* parTickCinfo = initParTickCinfo();

static const Slot processSlot = 
	initParTickCinfo()->getSlot( "process.process" );
static const Slot reinitSlot = 
	initParTickCinfo()->getSlot( "process.reinit" );

static const Slot outgoingProcessSlot = 
	initParTickCinfo()->getSlot( "outgoingProcess.process" );
static const Slot outgoingReinitSlot = 
	initParTickCinfo()->getSlot( "outgoingProcess.reinit" );

static const Slot iRecvSlot = 
	initParTickCinfo()->getSlot( "parTick.postIrecv" );
static const Slot sendSlot = 
	initParTickCinfo()->getSlot( "parTick.postSend" );
static const Slot pollSlot = 
	initParTickCinfo()->getSlot( "parTick.poll" );
static const Slot barrierSlot = 
	initParTickCinfo()->getSlot( "parTick.barrier" );
static const Slot clearSetupStackSlot = 
	initParTickCinfo()->getSlot( "parTick.clearSetupStack" );

///////////////////////////////////////////////////
// Field function definitions
///////////////////////////////////////////////////
/**
 * This is called to set the barrier flag. When it is true, then the
 * ParTick terminates only when an MPI_barrier is crossed. This
 * requires all nodes to cross the Barrier.
 */
void ParTick::setBarrier( const Conn* c, int v )
{
	
	static_cast< ParTick* >( c->data() )->barrier_ = ( v != 0 );
}

/**
 * The getStage just looks up the local stage, much less involved than
 * the setStage function.
 */
int ParTick::getBarrier( Eref e )
{
	return static_cast< ParTick* >( e.data() )->barrier_;
}

/**
 * This is called to set the barrier flag. When it is true, then the
 * ParTick terminates only when an MPI_barrier is crossed. This
 * requires all nodes to cross the Barrier.
 */
void ParTick::setSync( const Conn* c, bool v )
{
	static_cast< ParTick* >( c->data() )->doSync_ = v;
}

/**
 * The getStage just looks up the local stage, much less involved than
 * the setStage function.
 */
bool ParTick::getSync( Eref e )
{
	return static_cast< ParTick* >( e.data() )->doSync_;
}


///////////////////////////////////////////////////
// Dest function definitions
///////////////////////////////////////////////////

void ParTick::pollFunc( const Conn* c, unsigned int node ) 
{
	static_cast< ParTick* >( c->data() )->innerPollFunc( node );
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

void ParTick::printPos( const string& s )
{
	for ( unsigned int i = 0; i < Shell::myNode(); ++i )
		cout << "	";
	cout << s << doSync_ << endl << flush;
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
void ParTick::innerProcessFunc( Eref e, ProcInfo info )
{ 
#ifdef PRINT_POS
	printPos( "irc" );
	send0( e, iRecvSlot );
	// Phase 1: call Process for objects connected off-node
	printPos( "op1" );
	send1< ProcInfo >( e, outgoingProcessSlot, info );
	// Phase 2: send data off node
	printPos( "snd" );
	send1< bool >( e, sendSlot, doSync_ );
	// Phase 3: Call regular process for locally connected objects
	printPos( "op2" );
	send1< ProcInfo >( e, processSlot, info );
	// Phase 4: Poll for arrival of all data
	initPending( e );
	printPos( "pol" );
	while( pendingData() ) {
		send1< bool >( e, pollSlot, doSync_ );
	}
#else
	// Phase 6: execute barrier to sync all nodes, but only in sim mode.
	if ( doSync_ )  {
		// sendTo0( e, barrierSlot, 0 );
	}
	// Phase 0: post iRecv
	send0( e, iRecvSlot );
	// Phase 1: call Process for objects connected off-node
	send1< ProcInfo >( e, outgoingProcessSlot, info );
	// Phase 2: send data off node
	send1< bool >( e, sendSlot, doSync_ );
	// Phase 3: Call regular process for locally connected objects
	send1< ProcInfo >( e, processSlot, info );
	// Phase 4: Poll for arrival of all data
	initPending( e );
	while( pendingData() ) {
		send1< bool >( e, pollSlot, doSync_ );
	}
#endif

	// Phase 5: Clear all shell setup commands that have piled up.
	send0( e, clearSetupStackSlot );

}

void ParTick::initPending( Eref e )
{
	const Msg* m = e.e->msg( pollSlot.msg() );
	pendingCount_ = m->numTargets( e.e ) - 1;
	// cout << "pendingCount = " << pendingCount_ << endl;
	pendingNodes_.resize( pendingCount_ + 1 );
	pendingNodes_.assign( pendingCount_ + 1, 1 );
}

/**
 * Similar virtual function to deal with managing reinit info going out
 * to other nodes.
 */ 
void ParTick::innerReinitFunc( Eref e, ProcInfo info )
{
	/*
	*/
	if ( doSync_ ) {
#ifdef PRINT_POS
		printPos( e->name() + "reinbar" );
#endif
		sendTo0( e, barrierSlot, 0 );
	}
#ifdef PRINT_POS
		send0( e, iRecvSlot );
		printPos( e->name() + "reni0" );
		send1< ProcInfo >( e, outgoingReinitSlot, info );
		// Phase 2: send data off node
		printPos( e->name() + "rsnd" );
		send1< bool >( e, sendSlot, doSync_ );
		// Phase 3: Call regular reinit for locally connected objects
		printPos( e->name() + "reni1" );
		send1< ProcInfo >( e, reinitSlot, info );
		// Phase 4: Poll for arrival of all data
		initPending( e );
		printPos( e->name() + "rpol" );
		while( pendingData() )
			send1< bool >( e, pollSlot, doSync_ );
	
		// Phase 5: Clear all shell setup commands that have piled up.
		printPos( e->name() + "doop" );
		send0( e, clearSetupStackSlot );
#else
	// Here we need to scan all managed objects and decide if they
	// need to be on the outgoing process list or if they are local.
	//
	// separateOutgoingTargets
	//
	// if ( doSync_ ) sendTo0( e, barrierSlot, 0 );

//	if ( numOutgoing_ > 0 ) {
		// Phase 0: post iRecv
		// printPos( e->name() + "rirc" );
		send0( e, iRecvSlot );
		// Phase 1: call Reinit for objects connected off-node
		// printPos( e->name() + "reni0" );
		send1< ProcInfo >( e, outgoingReinitSlot, info );
		// Phase 2: send data off node
		// printPos( e->name() + "rsnd" );
		send1< bool >( e, sendSlot, doSync_ );
		// Phase 3: Call regular reinit for locally connected objects
		// printPos( e->name() + "reni1" );
		send1< ProcInfo >( e, reinitSlot, info );
		// Phase 4: Poll for arrival of all data
		initPending( e );
		// printPos( e->name() + "rpol" );
		while( pendingData() )
			send1< bool >( e, pollSlot, doSync_ );
	
		// Phase 5: Clear all shell setup commands that have piled up.
		send0( e, clearSetupStackSlot );
	/*
	} else {
		// Phase 3: Call regular reinit for locally connected objects
		// printPos( e->name() + "reni1" );
		send1< ProcInfo >( e, reinitSlot, info );
	}
	*/
#endif
}

/** 
 * Iterate through all process and outgoingProcess targets, 
 * make up a vector, then sort through and reassign depending on 
 * whether they have off-node messages to worry about.  
 *
 * Current version doesn't know what to do about array msgs.
 *
 */
void ParTick::innerResched( const Conn* c )
{
	updateNextTickTime( c->target() ); // sort out tick sequencing
	
	
	Eref tick = c->target();

	vector< Eref > targets;
	vector< int > targetMsgs;

	const Finfo* procFinfo = tick->findFinfo( "process" );
	const Finfo* outgoingProcFinfo = tick->findFinfo( "outgoingProcess" );
	assert( procFinfo != 0 );
	assert( outgoingProcFinfo != 0 );

	/////////////////////////////////////////////////////////////////////
	// Back up the target list
	/////////////////////////////////////////////////////////////////////
	Conn* i = tick->targets( procFinfo->msg(), tick.i );
	for ( ; i->good(); i->increment() ) {
		targets.push_back( i->target() );
		targetMsgs.push_back( i->targetMsg() );
	}
	delete i;

	i = tick->targets( outgoingProcFinfo->msg(), tick.i );
	for ( ; i->good(); i->increment() ) {
		targets.push_back( i->target() );
		targetMsgs.push_back( i->targetMsg() );
	}
	delete i;

	/////////////////////////////////////////////////////////////////////
	// Clean up old messages
	/////////////////////////////////////////////////////////////////////
	Msg* procMsg = tick->varMsg( procFinfo->msg() );
	Msg* outgoingProcMsg = tick->varMsg( outgoingProcFinfo->msg() );

	procMsg->dropAll( tick.e );
	outgoingProcMsg->dropAll( tick.e );
	numOutgoing_ = 0;

	/////////////////////////////////////////////////////////////////////
	// Build new messages, checking for off-node msgs.
	/////////////////////////////////////////////////////////////////////
	
	assert( targets.size() == targetMsgs.size() );
	Element* post = Id::postId( 0 ).eref().e;
	assert ( post != 0 );
	for ( unsigned int j = 0; j < targets.size(); j++ ) {
		Eref& tgt = targets[j];
		const Finfo* destFinfo = tgt->findFinfo( targetMsgs[j] );
		///\todo: to test, just connect up all objects to outgoing.
		// Later do it right and use all processed msgs.
		bool ret = 0;
		if ( tgt->isTarget( post ) ) {
			ret = outgoingProcFinfo->add( tick, tgt, destFinfo, 
				ConnTainer::Default );
			++numOutgoing_;
		} else {
			ret = procFinfo->add( tick, tgt, destFinfo, 
				ConnTainer::Default );
		}
		assert( ret );
	}
	// cout << c->target()->name() << "@" << Shell::myNode() << ": numOutgoing = " << numOutgoing_ << ", numTgts=" << targets.size() << endl << flush;
}


///////////////////////////////////////////////////
// Other function definitions
///////////////////////////////////////////////////
/// virtual function to start up ticks.  May need to set barrier.
void ParTick::innerStart( Eref e, ProcInfo p, double maxTime )
{
	if ( doSync_ ) {
#ifdef PRINT_POS
		printPos( "stbar" );
#endif
		sendTo0( e, barrierSlot, 0 );
	}

	Tick::innerStart( e, p, maxTime );
}