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
#include "TickMgr.h"
#include "TickPtr.h"
#include "ThreadInfo.h"
#include "Clock.h"

static const unsigned int OkStatus = ~0; // From Shell.cpp

/// Microseconds to sleep when not processing.
static const unsigned int SleepyTime = 50000; 


	///////////////////////////////////////////////////////
	// MsgSrc definitions
	///////////////////////////////////////////////////////
static SrcFinfo0 tickSrc( 
		"childTick",
		"Parent of Tick element"
	);

static SrcFinfo0 finished( 
		"finished",
		"Signal for completion of run"
	);

static SrcFinfo2< unsigned int, unsigned int > ack( 
		"ack",
		"Acknowledgement signal for receipt/completion of function."
		"Goes back to Shell on master node"
	);

const Cinfo* Clock::initCinfo()
{
	///////////////////////////////////////////////////////
	// Field definitions
	///////////////////////////////////////////////////////
		static ValueFinfo< Clock, double > runTime( 
			"runTime",
			"Duration to run the simulation",
			&Clock::setRunTime,
			&Clock::getRunTime
		);
		static ReadOnlyValueFinfo< Clock, double > currentTime(
			"currentTime",
			"Current simulation time",
			&Clock::getCurrentTime
		);
		static ValueFinfo< Clock, unsigned int > nsteps( 
			"nsteps",
			"Number of steps to advance the simulation, in units of the smallest timestep on the clock ticks",
			&Clock::setNsteps,
			&Clock::getNsteps
		);
		static ReadOnlyValueFinfo< Clock, unsigned int > numTicks( 
			"numTicks",
			"Number of clock ticks",
			// &Clock::setNumTicks,
			&Clock::getNumTicks
		);
		static ReadOnlyValueFinfo< Clock, unsigned int > currentStep( 
			"currentStep",
			"Current simulation step",
			&Clock::getCurrentStep
		);
	///////////////////////////////////////////////////////
	// Shared definitions
	///////////////////////////////////////////////////////

	///////////////////////////////////////////////////////
	// MsgDest definitions
	///////////////////////////////////////////////////////
		static DestFinfo start( "start", 
			"Sets off the simulation for the specified duration",
			new OpFunc1< Clock, double >(&Clock::handleStart )
		);

		static DestFinfo step( "step", 
			"Sets off the simulation for the specified # of steps",
			new EpFunc1< Clock, unsigned int >(&Clock::step )
		);

		static DestFinfo stop( "stop", 
			"Halts the simulation, with option to restart seamlessly",
			new EpFunc0< Clock >(&Clock::stop )
		);

		static DestFinfo setupTick( "setupTick", 
			"Sets up a specific clock tick: args tick#, dt",
			new OpFunc2< Clock, unsigned int, double >(&Clock::setupTick )
		);

		static DestFinfo reinit( "reinit", 
			"Zeroes out all ticks, starts at t = 0",
	// 		new EpFunc0< Clock >(&Clock::reinit )
	 		new OpFunc0< Clock >(&Clock::handleReinit )
		);
		static Finfo* clockControlFinfos[] = {
			&start, &step, &stop, &setupTick, &reinit, &ack,
		};
	///////////////////////////////////////////////////////
	// SharedFinfo for Shell to control Clock
	///////////////////////////////////////////////////////
		static SharedFinfo clockControl( "clockControl",
			"Controls all scheduling aspects of Clock, usually from Shell",
			clockControlFinfos, 
			sizeof( clockControlFinfos ) / sizeof( Finfo* )
		);
	///////////////////////////////////////////////////////
	// FieldElementFinfo definition for ticks.
	///////////////////////////////////////////////////////
		static FieldElementFinfo< Clock, Tick > tickFinfo( "tick",
			"Sets up field Elements for Tick",
			Tick::initCinfo(),
			&Clock::getTick,
			&Clock::setNumTicks,
			&Clock::getNumTicks
		);

	static Finfo* clockFinfos[] =
	{
		// Fields
		&runTime,
		&currentTime,
		&nsteps,
		&numTicks,
		&currentStep,
		// SrcFinfos
		&tickSrc,
		&finished,
		// DestFinfos
		/*
		&start,
		&step,
		&stop,
		&setupTick,
		&reinit,
		*/
		// Shared Finfos
		&clockControl,
		// FieldElementFinfo
		&tickFinfo,
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
		Neutral::initCinfo(),
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
	  doingReinit_( 0 ),
	  info_(),
	  numPendingThreads_( 0 ),
	  numThreads_( 0 ),
	  ticks_( Tick::maxTicks ),
		countNull1_( 0 ),
		countNull2_( 0 ),
		countReinit1_( 0 ),
		countReinit2_( 0 ),
		countAdvance1_( 0 ),
		countAdvance2_ ( 0 )
{
	for ( unsigned int i = 0; i < Tick::maxTicks; ++i ) {
		ticks_[i].setIndex( i );
	}
}
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
	return currentTime_;
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

///////////////////////////////////////////////////
// Dest function definitions
///////////////////////////////////////////////////

// After reinit, nextTime for everything is set to its own dt.
// Work out time for next clock, run until that
void Clock::start(  const Eref& e, const Qinfo* q, double runTime )
{
	static const double ROUNDING = 1.0000000001;
	if ( tickPtr_.size() == 0 ) {
		info_.currTime += runTime;
		return;
	}
	info_.currTime = tickPtr_[0].mgr()->getNextTime() - tickPtr_[0].mgr()->getDt();
	double endTime = runTime * ROUNDING + info_.currTime;
	isRunning_ = 1;

	/// Should actually use a lookup for child, or even postCreate Id.
	// Element* ticke = Id( 2 )();
	Id tickId( e.element()->id().value() + 1 );
	Element* ticke = tickId();

	if ( tickPtr_.size() == 1 ) {
		tickPtr_[0].mgr()->advance( ticke, &info_, endTime );
		return;
	}

	// Here we have multiple tick times, need to do the sorting.
	sort( tickPtr_.begin(), tickPtr_.end() );
	double nextTime = tickPtr_[1].mgr()->getNextTime();
	while ( isRunning_ && tickPtr_[0].mgr()->getNextTime() < endTime ) {
		// This advances all ticks with this dt in order, till nextTime.
		tickPtr_[0].mgr()->advance( ticke, &info_, nextTime * ROUNDING );
		sort( tickPtr_.begin(), tickPtr_.end() );
		nextTime = tickPtr_[1].mgr()->getNextTime();
	} 

	// Just to test: need to move back into the ticks:
	Qinfo::clearQ( &info_ );
	Qinfo::emptyAllQs();

	isRunning_ = 0;
}

void Clock::step(  const Eref& e, const Qinfo* q, unsigned int nsteps )
{
	double endTime = info_.currTime + dt_ * nsteps;
	sort( tickPtr_.begin(), tickPtr_.end() );
	start( e, q, endTime );
}

/**
 * Does a graceful stop of the simulation, leaving so it can continue
 * cleanly with another step or start command.
 */
void Clock::stop(  const Eref& e, const Qinfo* q )
{
	isRunning_ = 0;
}

/**
 * Does a disgraceful stop of the simulation, leaving it wherever it was.
 * Cannot resume.
 */
void Clock::terminate(  const Eref& e, const Qinfo* q )
{
	isRunning_ = 0; // Later we will be more vigourous about killing it.
}

/**
 * Reinit is used to reinit the state of the scheduling system.
 * Should be done single-threaded.
 */
void Clock::reinit( const Eref& e, const Qinfo* q )
{
	info_.currTime = 0.0;
	runTime_ = 0.0;
	currentTime_ = 0.0;
	nextTime_ = 0.0;
	nSteps_ = 0;
	currentStep_ = 0;

	Eref ticker( Id( 2 )(), 0 );
	for ( vector< TickPtr >::iterator i = tickPtr_.begin();
		i != tickPtr_.end(); ++i )
		i->mgr()->reinit( ticker, &info_ );
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
 * This function sets up a new tick, or reassigns an existing one.
 */
void Clock::setupTick( unsigned int tickNum, double dt )
{
	assert( tickNum < Tick::maxTicks );
	ticks_[ tickNum ].setDt( dt );
	// ticks_[ tickNum ].setStage( stage );
	rebuild();
}

///////////////////////////////////////////////////
// Other function definitions
///////////////////////////////////////////////////

void Clock::addTick( Tick* t )
{
	static const double EPSILON = 1.0e-9;

	if ( t->getDt() < EPSILON )
		return;
	for ( vector< TickMgr >::iterator j = tickMgr_.begin(); 
		j != tickMgr_.end(); ++j)
	{
		if ( j->addTick( t ) )
			return;
	}
	tickMgr_.push_back( t );
	// This is messy. The push_back has invalidated all earlier pointers.
	tickPtr_.clear();
	for ( vector< TickMgr >::iterator j = tickMgr_.begin(); 
		j != tickMgr_.end(); ++j)
	{
		tickPtr_.push_back( TickPtr( &( *j ) ) );
	}
}

void Clock::rebuild()
{
	Element* ticke = Id( 2 )();
	for ( unsigned int i = 0; i < Tick::maxTicks; ++i ) {
		ticks_[i].setIndex( i );
		ticks_[i].setElement( ticke );
	}

	tickPtr_.clear();
	tickMgr_.clear();
	for( unsigned int i = 0; i < ticks_.size(); ++i ) {
		addTick( &( ticks_[i] ) ); // This fills in only ticks that are used
	}
	sort( tickPtr_.begin(), tickPtr_.end() );
}


///////////////////////////////////////////////////
// These are the new scheduling base functions. 
// Here the clock is advanced one step on each cycle of the main loop.
// This means that a single tick advances one step.
// The clock advance is whatever is the minimum messaging interval.
// For typical simulations this is around 1 to 5 msec. The idea is that
// the major solvers will do their internal updates for this period, and
// their data interchange can be on this slower timescale. Still longer
// intervals come out from the slower ticks.
//
// The sequence of events is:
// processPhase1
// Barrier1
// processPhase2
// clearQ
// Barrier2
// MPI clearQ
// Barrier 3
///////////////////////////////////////////////////

bool Clock::keepLooping() const
{
	return keepLooping_;
}

void Clock::setLoopingState( bool val )
{
	keepLooping_ = val;
}

/**
 * The processPhase1 operation is called on every thread in the main event 
 * loop, during phase1 of the loop. This has to drive thread-specific 
 * calculations on all scheduled objects.
 */
void Clock::processPhase1( ProcInfo* info )
{
	if ( isRunning_ )
		advancePhase1( info );
	else if ( doingReinit_ )
		reinitPhase1( info );
	else if ( info->threadIndexInGroup == 0 )
		++countNull1_;
}

void Clock::processPhase2( ProcInfo* info )
{
	if ( isRunning_ )
		advancePhase2( info );
	else if ( doingReinit_ )
		reinitPhase2( info );
	else if ( info->threadIndexInGroup == 0 )
		++countNull2_;
}

/////////////////////////////////////////////////////////////////////
// Scheduling the 'process' operation for scheduled objects.
// Three functions are involved: the handler for the start function,
// the advancePhase1 and advancePhase2.
/////////////////////////////////////////////////////////////////////

/**
 * This has to happen on a single thread, whatever the Clock is assigned to.
 * Start has to happen gracefully: If the simulation was stopped for any
 * reason, it has to pick up where it left off.
 * runtime_ is the additional time to run the simulation. This is a little
 * odd when the simulation has stopped halfway through a clock tick.
 */
void Clock::handleStart( double runtime )
{
	static const double ROUNDING = 0.9999999999;
	if ( isRunning_ ) {
		cout << "Clock::handleStart: Warning: simulation already in progress.\n Command ignored\n";
		return;
	}
	if ( tickPtr_.size() == 0 || tickPtr_[0].mgr() == 0 ) {
		cout << "Clock::handleStart: Warning: simulation not yet initialized.\n Command ignored\n";
		return;
	}
	runTime_ = runtime;
	endTime_ = runtime * ROUNDING + currentTime_;
	isRunning_ = 1;
}

// Advance system state by one clock tick. This may be a subset of
// one timestep, as there may be multiple clock ticks within one dt.
// This simply distributes the call to all scheduled objects
void Clock::advancePhase1(  ProcInfo *p )
{
	tickPtr_[0].mgr()->advancePhase1( p );
	if ( p->threadIndexInGroup == 0 ) {
		++countAdvance1_;
	}
}

// In phase 2 we need to do the updates to the Clock object, especially
// sorting the TickPtrs. This also is when we find out if the simulation
// is finished.
void Clock::advancePhase2(  ProcInfo *p )
{
	if ( p->threadIndexInGroup == 0 ) {
		tickPtr_[0].mgr()->advancePhase2( p );
		if ( tickPtr_.size() > 1 )
			sort( tickPtr_.begin(), tickPtr_.end() );
		currentTime_ = tickPtr_[0].mgr()->getNextTime();
		if ( currentTime_ > endTime_ ) {
			Id clockId( 1 );
			isRunning_ = 0;
			finished.send( clockId.eref(), p );
			ack.send( clockId.eref(), p, p->nodeIndexInGroup, OkStatus );
		}
		++countAdvance2_;
	}
}

/////////////////////////////////////////////////////////////////////
// Scheduling the 'reinit' operation for scheduled objects.
// Three functions are involved: the handler for the reinit function,
// and the reinitPhase1 and reinitPhase2.
/////////////////////////////////////////////////////////////////////

/**
 * This is the dest function that sets off the reinit.
 */
void Clock::handleReinit()
{
	info_.currTime = 0.0;
	runTime_ = 0.0;
	currentTime_ = 0.0;
	nextTime_ = 0.0;
	nSteps_ = 0;
	currentStep_ = 0;
	isRunning_ = 0;
	doingReinit_ = 1;
}


/**
 * Reinit is used to reinit the state of the scheduling system.
 * This version is meant to be done through the multithread scheduling
 * loop.
 * In phase1 it calls reinit on all target Elements.
 */
void Clock::reinitPhase1( ProcInfo* info )
{
	for ( vector< TickPtr >::const_iterator i = tickPtr_.begin();
		i != tickPtr_.end(); ++i ) {
		i->mgr()->reinitPhase1( info );
	}
	if ( info->threadIndexInGroup == 0 )
		++countReinit1_;
}

/**
 * In phase2 it initializes internal TickMgr state variables.
 */
void Clock::reinitPhase2( ProcInfo* info )
{
	info->currTime = 0.0;
	if ( info->threadIndexInGroup == 0 ) {
		doingReinit_ = 0;
		for ( vector< TickPtr >::iterator i = tickPtr_.begin();
			i != tickPtr_.end(); ++i ) {
			i->mgr()->reinitPhase2( info );
		}
		sort( tickPtr_.begin(), tickPtr_.end() );
		Id clockId( 1 );
		ack.send( clockId.eref(), info, info->nodeIndexInGroup, OkStatus );
	}
	if ( info->threadIndexInGroup == 0 )
		++countReinit2_;
}


void Clock::printCounts() const
{
	cout << "Phase 1: Reinit = " << countReinit1_ <<
		";	advance = " << countAdvance1_ <<
		";	null = " << countNull1_ << endl;
	cout << "Phase 2: Reinit = " << countReinit2_ <<
		";	advance = " << countAdvance2_ <<
		";	null = " << countNull2_ << endl;
}
