/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2009 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

/**
 * The Clock manages simulation scheduling, in a close
 * collaboration with the Tick.
 * This new version does this using an array of child Ticks, which
 * it manages directly. This contrasts with the distributed approach
 * in the earlier version of MOOSE, where the clock ticks were more
 * independent and kept things organized using messages.
 * The reasons are a) to keep it simpler and b) because messages
 * are now themselves handled through queueing, and the management of
 * scheduling needs immediate function calls.
 *
 * Simulation scheduling requires that certain functions of 
 * groups of objects be called in a strict sequence according to
 * a range of timesteps, dt. For example, the numerical integration
 * function of a compartment may be called every 10 microseconds.
 * Furthermore, there may be ion channels in the simulation which also
 * have to be called every 10 microseconds, but always after all
 * the compartments have been called. Finally,
 * the graphical update function to plot the compartment voltage
 * may be called every 1 millisecond of simulation time.
 * The whole sequence has to be repeated till the runtime of the 
 * simulation is complete.
 *
 * In addition to all this, the scheduler has to interleave between
 * 'process' and 'clearQ' calls to target Elements. Furthermore, the
 * scheduler still has to keep 'clearQ' calls going when the processing
 * is not running, for setup.
 *
 * This scheduler is quite general and handles any combination of
 * simulation times, including non-multiple ratios of dt.
 *
 * The Clock part of the team manages a set of Ticks.
 * Each Tick handles a given dt and a given stage within a dt.
 * The scheduler guarantees that the call sequence is preserved
 * between Ticks, but there are no sequence assumptions within
 * a single Tick.
 *
 * The system works like this:
 * 1. We create a bunch of Ticks on the Clock using addTick
 * 		This directly sorts and assigns reasonable starting values.
 * 2. We assign their dts and stages within each dt, if needed.
 * 3. We connect up the Ticks to their target objects.
 * 4. We call Resched on the Clock... Not needed. Reinit is OK.
 * 5. We begin the simulation by calling 'start' or 'step' on the Clock.
 * 6. To interrupt the simulation at some intermediate time, call 'halt'.
 * 7. To restart the simulation from where it left off, use the same 
 * 		'start' or 'step' function on the Clock. As all the ticks
 * 		retain their state, the simulation can resume smoothly.
 */

#include <pthread.h>
#include "header.h"
#include "Tick.h"
#include "TickPtr.h"
#include "Clock.h"

const unsigned int tickSlot = 0;
static SrcFinfo0* tickSrc = 
	new SrcFinfo0( 
		"tick",
		"Parent of Tick element",
		tickSlot
	);

const unsigned int finishedSlot = 1;
static SrcFinfo0* finished = 
	new SrcFinfo0( 
		"finished",
		"Signal for completion of run",
		finishedSlot
	);

const Cinfo* Clock::initCinfo()
{
	static Finfo* clockFinfos[] =
	{
	///////////////////////////////////////////////////////
	// Field definitions
	///////////////////////////////////////////////////////
		new ValueFinfo< Clock, double >( 
			"runTime",
			"Duration to run the simulation",
			&Clock::setRunTime,
			&Clock::getRunTime
		),
		new ReadonlyValueFinfo< Clock, double >(
			"currentTime",
			"Current simulation time",
			&Clock::getCurrentTime
		),
		new ValueFinfo< Clock, unsigned int >( 
			"nsteps",
			"Number of steps to advance the simulation, in units of the smallest timestep on the clock ticks",
			&Clock::setNsteps,
			&Clock::getNsteps
		),
		new ValueFinfo< Clock, unsigned int >( 
			"numTicks",
			"Number of clock ticks",
			&Clock::setNumTicks,
			&Clock::getNumTicks
		),
		new ValueFinfo< Clock, unsigned int >( 
			"numPendingThreads",
			"Number of threads still to update",
			&Clock::setNumPendingThreads,
			&Clock::getNumPendingThreads
		),
		new ValueFinfo< Clock, unsigned int >( 
			"numThreads",
			"Number of threads",
			&Clock::setNumThreads,
			&Clock::getNumThreads
		),
		new ReadonlyValueFinfo< Clock, unsigned int >( 
			"currentStep",
			"Current simulation step",
			&Clock::getCurrentStep
		),
	///////////////////////////////////////////////////////
	// Shared definitions
	///////////////////////////////////////////////////////
	///////////////////////////////////////////////////////
	// MsgSrc definitions
	///////////////////////////////////////////////////////
		tickSrc,
		finished,

	///////////////////////////////////////////////////////
	// MsgDest definitions
	///////////////////////////////////////////////////////
		new DestFinfo( "start", 
			"Sets off the simulation for the specified duration",
			new EpFunc1< Clock, double >(&Clock::start )
		),

		new DestFinfo( "step", 
			"Sets off the simulation for the specified # of steps",
			new EpFunc1< Clock, unsigned int >(&Clock::step )
		),

		new DestFinfo( "stop", 
			"Halts the simulation, with option to restart seamlessly",
			new EpFunc0< Clock >(&Clock::stop )
		),

		new DestFinfo( "setupTick", 
			"Sets up a specific clock tick: args tick#, dt, stage",
			new OpFunc3< Clock, unsigned int, double, unsigned int >(&Clock::setupTick )
		),

		new DestFinfo( "reinit", 
			"Zeroes out all ticks, starts at t = 0",
			new EpFunc0< Clock >(&Clock::reinit )
		),
	};
	
	static string doc[] =
	{
		"Name", "Clock",
		"Author", "Upinder S. Bhalla, Mar 2007, NCBS",
		"Description", "Clock: Clock class. Handles sequencing of operations in simulations",
	};

	static Cinfo clockCinfo(
		"Clock",
		// "Clock class handles sequencing of operations in simulations",
		0,
		clockFinfos,
		sizeof(clockFinfos)/sizeof(Finfo *),
		new Dinfo< Clock >()
	);

	return &clockCinfo;
}

static const Cinfo* clockCinfo = Clock::initCinfo();

///////////////////////////////////////////////////
// Constructor
///////////////////////////////////////////////////
Clock::Clock()
	: runTime_( 0.0 ), 
	  currentTime_( 0.0 ), 
	  nextTime_( 0.0 ),
	  nSteps_( 0 ), 
	  currentStep_( 0 ), 
	  dt_( 1.0 ), 
	  isRunning_( 0 ),
	  info_(),
	  numPendingThreads_( 0 ),
	  numThreads_( 0 )
{;}
///////////////////////////////////////////////////
// Field function definitions
///////////////////////////////////////////////////

/**
 * This sets the runtime to use for the simulation.
 * Perhaps this should not be assignable, but readonly.
 * Or The assignment should be a message.
 */
void Clock::setRunTime( double v )
{
	runTime_ = v;
}
double Clock::getRunTime() const
{
	return runTime_;
}

double Clock::getCurrentTime() const
{
	return info_.currTime;
}

/**
 * This sets the number of steps to use for the simulation.
 * Perhaps this should not be assignable, but readonly.
 * Or The assignment should be a message.
 * It is a variant of the runtime function.
 */
void Clock::setNsteps( unsigned int v )
{
	nSteps_ = v;
}
unsigned int Clock::getNsteps() const
{
	return nSteps_;
}

unsigned int Clock::getCurrentStep() const
{
	return currentStep_;
}

Tick* Clock::getTick( unsigned int i )
{
	if ( i < ticks_.size() )
		return &ticks_[i];
	else
		return 0;
}

unsigned int Clock::getNumTicks() const
{
	return ticks_.size();
}

void Clock::setNumTicks( unsigned int num )
{
	ticks_.resize( num );
	rebuild();
}

Element* Clock::getTickE( Element* e )
{
	const Conn* c = e->conn( tickSlot );
	assert( c != 0 );
	Element* ret = c->getTargetElement( e, 0 );
	assert( ret );
	return ret;
}

unsigned int Clock::getNumPendingThreads() const
{
	return numPendingThreads_;
}

void Clock::setNumPendingThreads( unsigned int num )
{
	numPendingThreads_ = num;
}

unsigned int Clock::getNumThreads() const
{
	return numThreads_;
}

void Clock::setNumThreads( unsigned int num )
{
	numThreads_ = num;
	info_.numThreads = num;
}

///////////////////////////////////////////////////
// Dest function definitions
///////////////////////////////////////////////////

// After reinit, nextTime for everything is set to its own dt.
// Work out time for next clock, run until that
void Clock::start(  Eref e, const Qinfo* q, double runTime )
{
	static const double ROUNDING = 1.0000000001;
	if ( tickPtr_.size() == 0 ) {
		info_.currTime += runTime;
		return;
	}
	double endTime = runTime * ROUNDING + info_.currTime;
	isRunning_ = 1;

	// Element* ticke = getTickE( e.element() );
	Element* ticke = Id( 2, 0 )();

	if ( tickPtr_.size() == 1 ) {
		tickPtr_[0].advance( ticke, &info_, endTime, 0 );
		return;
	}

	// Here we have multiple tick times, need to do the sorting.
	sort( tickPtr_.begin(), tickPtr_.end() );
	double nextTime = tickPtr_[1].getNextTime();
	while ( isRunning_ && tickPtr_[0].getNextTime() < endTime ) {
		// This advances all ticks with this dt in order, till nextTime.
		tickPtr_[0].advance( ticke, &info_, nextTime * ROUNDING, 0 );
		sort( tickPtr_.begin(), tickPtr_.end() );
		nextTime = tickPtr_[1].getNextTime();
	}
	isRunning_ = 0;
}


// Problem: how do we set up 'numPendingThreads_'? Must do in the parent
// thread.
// Sets up the tick Ptrs as the first thread goes through. Resets the
// counting variable after all are through. Requires that there is a
// barrier or equivalent downstream, to ensure that any given thread
// doesn't come back to this function before all other threads are done.
void Clock::sortTickPtrs( pthread_mutex_t* sortMutex )
{
	pthread_mutex_lock( sortMutex );
		numPendingThreads_++;
		if ( numPendingThreads_ == 1 ) { // The first thread through should do this.
			sort( tickPtr_.begin(), tickPtr_.end() );
			nextTime_ = tickPtr_[1].getNextTime();
			tp0_ = &tickPtr_[ 0 ];
			// Someone else may modify isRunning_: don't touch.
		} 
		if ( numPendingThreads_ >= numThreads_ )
			numPendingThreads_ = 0;
	pthread_mutex_unlock( sortMutex );
	// cout << "Clock::sortTickPtrs: numPending = " << numPendingThreads_ << ", nextTime_ = " << nextTime_ << endl;
}

// This version uses sortTickPtrs rather than a bunch of barriers.
void Clock::tStart(  Eref e, const ThreadInfo* ti )
{
	ProcInfo pinfo = info_; //We use an independent ProcInfo for each thread
	pinfo.threadId = ti->threadId; // to manage separate threadIds.
	pinfo.threadIndexInGroup = ti->threadIndexInGroup;
	assert( ti->groupId < Qinfo::numSimGroup() );
	pinfo.numThreadsInGroup = Qinfo::simGroup( ti->groupId )->numThreads;
	pinfo.groupId = ti->groupId;
	pinfo.outQid = ti->outQid;
	assert( pinfo.numThreads == numThreads_ );
	static const double ROUNDING = 1.0000000001;
	if ( tickPtr_.size() == 0 ) {
		if ( ti->threadId == 0 )
			info_.currTime += ti->runtime;
		return;
	}
	info_.currTime = tickPtr_[0].getNextTime() - tickPtr_[0].getDt();
	double endTime = ti->runtime * ROUNDING + info_.currTime;
	isRunning_ = 1;

	// Element* ticke = getTickE( e.element() );
	Element* ticke = Id( 2, 0 )();

	if ( tickPtr_.size() == 1 ) {
		tickPtr_[0].advance( ticke, &pinfo, endTime, ti->timeMutex );
		if ( ti->threadId == 0 )
			info_ = pinfo; // Do we use info outside? Shouldn't.
		return;
	}

	sortTickPtrs( ti->sortMutex ); // Sets up nextTime_ and tp0_.
	
	while ( isRunning_ && tp0_->getNextTime() < endTime ) {
		// This advances all ticks with this dt in order, till nextTime.
		// It has a barrier within: essential for the sortTickPtrs to work.
		tp0_->advance( ticke, &pinfo, nextTime_ * ROUNDING, ti->timeMutex );
		// cout << "Advance at " << nextTime_ << " on thread " << ti->threadId << endl;
		sortTickPtrs( ti->sortMutex ); // Sets up nextTime_ and tp0_.
	}
	if ( ti->threadId == 0 )
		info_ = pinfo;
	isRunning_ = 0;
}

void Clock::setBarrier( void* barrier )
{
	info_.barrier = barrier;
}

// Static function used to pass into pthread_create
void* Clock::threadStartFunc( void* threadInfo )
{
	ThreadInfo* ti = reinterpret_cast< ThreadInfo* >( threadInfo );
	Clock* clock = reinterpret_cast< Clock* >( ti->clocke->data( 0 ) );
	Eref clocker( ti->clocke, 0 );
	// cout << "Start thread " << ti->threadId << " threadIndex in Group " << ti->threadIndexInGroup << endl;
	clock->tStart( clocker, ti );
	// cout << "End thread " << ti->threadId << " with runtime " << ti->runtime << endl;

	pthread_exit( NULL );
}

void Clock::step(  Eref e, const Qinfo* q, unsigned int nsteps )
{
	double endTime = info_.currTime + dt_ * nsteps;
	sort( tickPtr_.begin(), tickPtr_.end() );
	start( e, q, endTime );
}

/**
 * Does a graceful stop of the simulation, leaving so it can continue
 * cleanly with another step or start command.
 */
void Clock::stop(  Eref e, const Qinfo* q )
{
	isRunning_ = 0;
}

/**
 * Reinit is used to reinit the state of the scheduling system.
 * Should be done single-threaded.
 */
void Clock::reinit( Eref e, const Qinfo* q )
{
	info_.currTime = 0.0;
	runTime_ = 0.0;
	currentTime_ = 0.0;
	nextTime_ = 0.0;
	nSteps_ = 0;
	currentStep_ = 0;
	// more stuff.
	for ( vector< TickPtr >::iterator i = tickPtr_.begin();
		i != tickPtr_.end(); ++i )
		i->reinit( e );
}

/**
 * This function handles any changes to dt in the ticks. This means
 * it must redo the ordering of the ticks and call a resched on them.
 */
void Clock::setTickDt( DataId i, double dt )
{
	if ( i.field() < ticks_.size() ) {
		ticks_[ i.field() ].setDt( dt ); 
		rebuild();
	} else {
		cout << "Clock::setTickDt:: Tick " << i << " not found\n";
	}
}

double Clock::getTickDt( DataId i ) const
{
	if ( i.field() < ticks_.size() ) {
		return ticks_[ i.field() ].getDt(); 
	} else {
		cout << "Clock::getTickDt:: Tick " << i << " not found\n";
	}
	return 1.0;
}

/**
 * This function handles any changes to stage in the ticks. This means
 * it must redo the ordering of the ticks and call a resched on them.
 */
void Clock::setStage( DataId i, unsigned int val )
{
	if ( i.field() < ticks_.size() ) {
		ticks_[ i.field() ].setStage( val ); 
		rebuild();
	} else {
		cout << "Clock::setStage:: Tick " << i << " not found\n";
	}
}

unsigned int Clock::getStage( DataId i ) const
{
	if ( i.field() < ticks_.size() ) {
		return ticks_[ i.field() ].getStage(); 
	} else {
		cout << "Clock::getStage:: Tick " << i << " not found\n";
	}
	return 0;
}

/**
 * This function sets up a new tick, or reassigns an existing one.
 */
void Clock::setupTick( unsigned int tickNum, double dt, unsigned int stage )
{
	if ( tickNum >= ticks_.size() ) {
		ticks_.resize( tickNum + 1 );
	}
	ticks_[ tickNum ].setDt( dt );
	ticks_[ tickNum ].setStage( stage );
	rebuild();
}

///////////////////////////////////////////////////
// Other function definitions
///////////////////////////////////////////////////

void Clock::process( const ProcInfo* p, const Eref& e )
{
	;
}

void Clock::addTick( Tick* t )
{
	static const double EPSILON = 1.0e-9;

	if ( t->getDt() < EPSILON )
		return;
	for ( vector< TickPtr >::iterator j = tickPtr_.begin(); 
		j != tickPtr_.end(); ++j)
	{
		if ( j->addTick( t ) )
			return;
	}
	tickPtr_.push_back( t );
}

void Clock::rebuild()
{
	tickPtr_.clear();
	for( unsigned int i = 0; i < ticks_.size(); ++i ) {
		ticks_[i].setIndex( i );
		addTick( &( ticks_[i] ) );
	}
	/*
	for ( vector< Tick >::iterator i = ticks_.begin(); 
		i != ticks_.end(); ++i)
	{
		addTick( &( *i ) );
	}
	*/
	sort( tickPtr_.begin(), tickPtr_.end() );
}

