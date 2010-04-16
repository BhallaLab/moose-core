/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2009 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/
#include "header.h"
#include "Tick.h"
#include "TickPtr.h"
#include "Clock.h"

/**
 * The Tick handles the nuts and bolts of scheduling. It sends the
 * Process (or other event) message to all the scheduled objects,
 * and it keeps track of the update sequence with its sibling Tick
 * objects.
 */

// This vector contains the SrcFinfos used for Process calls for each
// of the Ticks.
vector< SrcFinfo* > process;

static SrcFinfo1< ProcPtr > proc0( "process0", "Process for Tick 0" );
static SrcFinfo1< ProcPtr > proc1( "process1", "Process for Tick 1" );
static SrcFinfo1< ProcPtr > proc2( "process2", "Process for Tick 2" );
static SrcFinfo1< ProcPtr > proc3( "process3", "Process for Tick 3" );
static SrcFinfo1< ProcPtr > proc4( "process4", "Process for Tick 4" );
static SrcFinfo1< ProcPtr > proc5( "process5", "Process for Tick 5" );
static SrcFinfo1< ProcPtr > proc6( "process6", "Process for Tick 6" );
static SrcFinfo1< ProcPtr > proc7( "process7", "Process for Tick 7" );
static SrcFinfo1< ProcPtr > proc8( "process8", "Process for Tick 8" );
static SrcFinfo1< ProcPtr > proc9( "process9", "Process for Tick 9" );

static SrcFinfo* procVec[] = {
	&proc0, &proc1, &proc2, &proc3, &proc4, &proc5, &proc6, &proc7, &proc8, &proc9, };

/*
SrcFinfo* makeProcessFinfo()
{
	static int i = 0;
	stringstream ss;
	ss << "process" << i;
	SrcFinfo* ret = 
		new SrcFinfo1< ProcPtr > (
			ss.str(),
			"Calls Process on target Elements. Indexed by Tick#",
		);
	process.push_back( ret );
	++i;
	return ret;
}
*/

const Cinfo* Tick::initCinfo()
{
	///////////////////////////////////////////////////////
	// Field definitions
	///////////////////////////////////////////////////////
		// Refers it to the parent Clock.
		static UpValueFinfo< Clock, double > dt(
			"dt",
			"Timestep for this tick",
			&Clock::setTickDt,
			&Clock::getTickDt
		);
		static ValueFinfo< Tick, double > localdt(
			"localdt",
			"Timestep for this tick",
			&Tick::setDt,
			&Tick::getDt
		);
		static UpValueFinfo< Clock, unsigned int > stage(
			"stage",
			"Sequence number if multiple ticks have the same dt.",
			&Clock::setStage,
			&Clock::getStage
		);
		static ValueFinfo< Tick, string> path(
			"path",
			"Wildcard path of objects managed by this tick",
			&Tick::setPath,
			&Tick::getPath
		);
	///////////////////////////////////////////////////////
	// MsgSrc definitions
	///////////////////////////////////////////////////////
		// These are defined outside this function.

	///////////////////////////////////////////////////////
	// MsgDest definitions
	///////////////////////////////////////////////////////
		static DestFinfo parent( "parent", 
			"Message from Parent Element(s)", 
			new EpFunc0< Tick >( &Tick::destroy ) );

	static Finfo* tickFinfos[] =
	{
		// Fields
		&dt,
		&localdt,
		&stage,
		&path,
		// SrcFinfos for process
		&proc0,
		&proc1,
		&proc2,
		&proc3,
		&proc4,
		&proc5,
		&proc6,
		&proc7,
		&proc8,
		&proc9,
		// MsgDest definitions
		&parent, // I thought this was to be inherited?
	};
	
	static Cinfo tickCinfo(
		"Tick",
		0,
		tickFinfos,
		sizeof(tickFinfos) / sizeof(Finfo *),
		new Dinfo< Tick >()
	);

	return &tickCinfo;
}

static const Cinfo* tickCinfo = Tick::initCinfo();

///////////////////////////////////////////////////
// Tick class definition functions
///////////////////////////////////////////////////
Tick::Tick()
	: dt_( 0.0 ), stage_( 0 )
{ ; }

Tick::~Tick()
{ ; }

bool Tick::operator<( const Tick& other ) const
{
	const double EPSILON = 1e-9;

	if ( dt_ < other.dt_ ) return 1;
		if ( fabs( 1.0 - dt_ / other.dt_ ) < EPSILON && 
			stage_ < other.stage_ )
			return 1;
	return 0;
}


bool Tick::operator==( const Tick& other ) const
{
	return ( dt_ == other.dt_ && stage_ == other.stage_ );
}

///////////////////////////////////////////////////
// Field function definitions
///////////////////////////////////////////////////

/**
 * This is called when dt is set on the local Tick.
 * We first fix up local values for nextTime and dt, then
 * we ask the parent ClockJob to re-sort the clock ticks to
 * put them back in order.
 * There should also be a bit here to set the parent.
 */
void Tick::setDt( double newdt )
{
	/*
	nextTime_ += newdt - dt_;
	if ( nextTime_ < dt_ )
		nextTime_ = dt_;
	*/
	dt_ = newdt;
}
/**
 * The getDt just looks up the local dt, much less involved than
 * the setDt function.
 */
double Tick::getDt() const
{
	return dt_;
}

/**
 * This is called when stage is set on the local Tick.
 * Like the setDt, it has to ask the parent ClockJob to
 * re-sort the clock ticks to put them back in order.
 */
void Tick::setStage( unsigned int v )
{
	stage_ = v;
}

/**
 * The getStage just looks up the local stage, much less involved than
 * the setStage function.
 */
unsigned int Tick::getStage() const
{
	return stage_;
}

/**
 * nextTime is here to peek into when the tick is due to fire next.
 * Not clear if it should become private.
double Tick::getNextTime() const
{
	// return nextTime_;
	return 0.0;
}
 */

/**
 * set and get Path are problematic. Their goal is to assign the 
 * targets for this Tick. As framed, they fit in with the older
 * GENESIS syntax. For now, put in dummy functions.
 */
void Tick::setPath( string v )
{
	path_ = v;
	// Something here to change the messages
}

string Tick::getPath() const
{
	return path_;
}

///////////////////////////////////////////////////
// Dest function definitions
///////////////////////////////////////////////////

void Tick::destroy( Eref e, const Qinfo* q )
{
	;
}


///////////////////////////////////////////////////
// Virtual function definitions for actually sending out the 
// process and reinit calls.
///////////////////////////////////////////////////

/**
 * This handles the mpi stuff.
 */
void Tick::mpiAdvance( ProcInfo* info) const
{
	// cout << info->nodeIndexInGroup << ", " << info->threadId << ": Tick::mpiAdvance (" << dt_ << ", " << stage_ << " ) at t= " << info->currTime << endl;
	assert( info->barrier );
	int rc = pthread_barrier_wait(
		reinterpret_cast< pthread_barrier_t* >( info->barrier ) );
	assert( rc == 0 || rc == PTHREAD_BARRIER_SERIAL_THREAD );
	// mergeQ is going on. Wait for inQ to be updated. Use time to clear mpiQ
	rc = pthread_barrier_wait(
		reinterpret_cast< pthread_barrier_t* >( info->barrier ) );
	// readQ is going on. InQ is ready. Do data transfer between nodes.
		assert( rc == 0 || rc == PTHREAD_BARRIER_SERIAL_THREAD );
		Qinfo::sendAllToAll( info );
	
	rc = pthread_barrier_wait(
		reinterpret_cast< pthread_barrier_t* >( info->barrier ) );
	assert( rc == 0 || rc == PTHREAD_BARRIER_SERIAL_THREAD );
	// readMpiQ is going on, and also process. Data has been transferred.
}

/**
 * This sends out the process call.
 */
void Tick::advance( Element* e, ProcInfo* info ) const
{
	
	assert( ( info->numNodesInGroup > 1 ) == ( info->numThreads == (info->numThreadsInGroup + 1) ) );
	// This is the mpiThread.
	if ( info->isMpiThread ) {
		mpiAdvance( info );
	} else {
		// cout << info->nodeIndexInGroup << ", " << info->threadId << ": Tick::advance (" << dt_ << ", " << stage_ << " ) at t= " << info->currTime << endl;

	/**
	 * This barrier pair protects the inQ from being accessed for reading, 
	 * while it is being updated, and vice versa.
	 */
	if ( info->barrier ) {
		int rc = pthread_barrier_wait(
			reinterpret_cast< pthread_barrier_t* >( info->barrier ) );
		assert( rc == 0 || rc == PTHREAD_BARRIER_SERIAL_THREAD );
	}
	// Start updating inQ
	// Have to ensure mpiThread does not do anything with inQ for a bit.
	if ( info->threadIndexInGroup == 0 ) {
		// Put the queues into one big one. Clear others
		// Qinfo::reportQ();
		Qinfo::mergeQ( info->groupId ); 
		// cout << "Tick::advance: t = " << info->currTime;
		// Send out all stuff in inQ to current group on other nodes
		// Harvest data for current node.
	}
		
	if ( info->barrier ) {
		int rc = pthread_barrier_wait(
			reinterpret_cast< pthread_barrier_t* >( info->barrier ) );
		assert( rc == 0 || rc == PTHREAD_BARRIER_SERIAL_THREAD );
	}
	// Set off mpiThread for MPI_alltoall data exchange of inQ
	// Start reading inQ
	Qinfo::readQ( info ); // March through inQ. Each thread magically deals
		// with updates needed by its own Process calls, and none other.

	if ( info->numNodesInGroup > 1 ) // Sync up with mpiThreadfunc
	{ // Sync up with mpiAdvance.
		// At this point the MPI_alltoall should have completed
		if ( info->barrier ) {
			int rc = pthread_barrier_wait(
				reinterpret_cast< pthread_barrier_t* >( info->barrier ) );
			assert( rc == 0 || rc == PTHREAD_BARRIER_SERIAL_THREAD );
		}
		Qinfo::readMpiQ( info ); // March through mpiQ
		// March through process calls
#if 0
		BindIndex b = procVec[ index_ ]->getBindIndex();
		const vector< MsgFuncBinding >* m = e->getMsgAndFunc( b );
		for ( vector< MsgFuncBinding >::const_iterator i = m->begin();
			i != m->end(); ++i )
			Msg::getMsg( i->mid )->process( info );
#endif
	}


	// March through Process calls for each scheduled Element.
	// Note that there is a unique BindIndex for each tick.
	// We preallocate 0 through 10 for this. May need to rethink.
	assert( index_ < sizeof( procVec ) / sizeof( SrcFinfo* ) );
	BindIndex b = procVec[ index_ ]->getBindIndex();
	const vector< MsgFuncBinding >* m = e->getMsgAndFunc( b );
	for ( vector< MsgFuncBinding >::const_iterator i = m->begin();
		i != m->end(); ++i )
		Msg::getMsg( i->mid )->process( info );
	}

}

void Tick::setIndex( unsigned int index ) 
{
	index_ = index;
}

/**
 * This sends out the call to reinit objects. It is virtualized
 * because derived classes (ParTick) need to do much more complicated
 * things to coordinate the reinit.
 */
void Tick::reinit( Eref e ) const
{
	;
	// nextTime_ = dt_;
	// send1< ProcInfo >( e, reinitSlot, info );
}
