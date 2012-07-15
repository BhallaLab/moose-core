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

#include "../shell/Shell.h"

static const unsigned int OkStatus = ~0; // From Shell.cpp

/// Microseconds to sleep when not processing.
static const unsigned int SleepyTime = 50000; 

/// Flag to tell Clock how to alter state gracefully in process loop.
Clock::ProcState Clock::procState_ = NoChange;

	///////////////////////////////////////////////////////
	// MsgSrc definitions
	///////////////////////////////////////////////////////
static SrcFinfo0 *tickSrc() {
	static SrcFinfo0 tickSrc( 
			"childTick",
			"Parent of Tick element"
			);
	return &tickSrc;
}

static SrcFinfo0 *finished() {
	static SrcFinfo0 finished( 
			"finished",
			"Signal for completion of run"
			);
	return &finished;
}

static SrcFinfo2< unsigned int, unsigned int > *ack() {
	static SrcFinfo2< unsigned int, unsigned int > ack( 
			"ack",
			"Acknowledgement signal for receipt/completion of function."
			"Goes back to Shell on master node"
			);
	return &ack;
}

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
			new OpFunc1< Clock, unsigned int >(&Clock::handleStep )
		);

		static DestFinfo stop( "stop", 
			"Halts the simulation, with option to restart seamlessly",
			new OpFunc0< Clock >(&Clock::stop )
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
			&start, &step, &stop, &setupTick, &reinit, ack(),
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
	// Setup max of 16 of them.
	///////////////////////////////////////////////////////
		static FieldElementFinfo< Clock, Tick > tickFinfo( "tick",
			"Sets up field Elements for Tick",
			Tick::initCinfo(),
			&Clock::getTick,
			&Clock::setNumTicks,
			&Clock::getNumTicks,
			16
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
		tickSrc(),
		finished(),
		// DestFinfos
		// Shared Finfos
		&clockControl,
		// FieldElementFinfo
		&tickFinfo,
	};
	
	static string doc[] =
	{
		"Name", "Clock",
		"Author", "Upinder S. Bhalla, Mar 2007, NCBS",
		"Description", "Clock: Clock class. Handles sequencing of operations in simulations."
		"Every object scheduled for operations in MOOSE is connected to one"
		"of the 'Tick' objects sitting as children on the Clock."
		"The Clock manages ten 'Ticks', each of which has its own dt."
		"The Ticks increment their internal time by their 'dt' every time "
		"they are updated, and in doing so they also call the Process"
		"function for every object that is connected to them."
		"The default scheduling (should not be overridden) has the "
		"following assignment of classes to Ticks:"
		"0: Biophysics - Init call on Compartments in EE method"
		"1: Biophysics - Channels"
		"2: Biophysics - Process call on Compartments"
		"3: ?"
		"4: Kinetics - Pools, or in ksolve mode: Mesh to handle diffusion"
		"5: Kinetics - Reacs, enzymes, etc, or in ksolve mode: Stoich/GSL"
		"6: Stimulus tables"
		"7: More stimulus tables"
		"8: Plots"
		"9: Slower graphics like cell arrays or 3-D displays"
		"",
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
	  currTickPtr_( 0 ),
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

/**
 * Does a graceful stop of the simulation, leaving so it can continue
 * cleanly with another step or start command.
 * This function can be called safely from any thread, provided it is
 * not within barrier3.
 */
void Clock::stop()
{
	procState_ = StopOnly;
}

/**
 * This function handles any changes to dt in the ticks. This means
 * it must redo the ordering of the ticks and call a resched on them.
 */
void Clock::setTickDt( unsigned int i, double dt )
{
	if ( i < ticks_.size() ) {
		ticks_[ i ].setDt( dt ); 
		rebuild();
	} else {
		cout << "Clock::setTickDt:: Tick " << i << " not found\n";
	}
}

double Clock::getTickDt( unsigned int i ) const
{
	if ( i < ticks_.size() ) {
		return ticks_[ i ].getDt(); 
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
	// ack()->send( clockId.eref(), p, p->nodeIndexInGroup, OkStatus );
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
	if ( tickPtr_.size() == 0 ) // Nothing happening in any of the ticks.
		return;

	// Here we put in current time so we can resume after changing a 
	// dt. Given that we are rebuilding and cannot
	// assume anything about prior dt, we do the simple thing and put
	// all the tickMgrs at the current time.
	for( vector< TickMgr >::iterator i = tickMgr_.begin(); 
		i != tickMgr_.end(); ++i ) {
		i->setNextTime( currentTime_ + i->getDt() );
	}

	sort( tickPtr_.begin(), tickPtr_.end() );
	dt_ = tickPtr_[0].mgr()->getDt();
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

bool Clock::isRunning() const
{
	return isRunning_;
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
	else if ( Shell::isSingleThreaded() || info->threadIndexInGroup == 1 )
		++countNull1_;
}

void Clock::processPhase2( ProcInfo* info )
{
	if ( isRunning_ )
		advancePhase2( info );
	else if ( doingReinit_ )
		reinitPhase2( info );
	else if ( Shell::isSingleThreaded() || info->threadIndexInGroup == 1 )
		++countNull2_;
}

/**
 * Static function, used to flip flags to start or end a simulation. 
 * It is used as the within-barrier function of barrier 3.
 * This has to be in the barrier as we are altering a Clock field which
 * the 'process' flag depends on.
 * Six cases:
 * 	- Do nothing
 * 	- Reinit only
 *	- Reinit followed by start
 *	- Start only
 *	- Stop only
 *	- Stop followed by Reinit.
 * Some of these cases need additional intermediate steps, since the reinit
 * flag has to turn itself off after one cycle.
 */
void Clock::checkProcState()
{
	/// Handle pending Reduce operations.
	Qinfo::clearReduceQ( Shell::numProcessThreads() );

	if ( procState_ == NoChange ) { // Most common 
		return;
	}

	Id clockId( 1 );
	assert( clockId() );
	Clock* clock = reinterpret_cast< Clock* >( clockId.eref().data() );

	switch ( procState_ ) {
		case TurnOnReinit:
			clock->doingReinit_ = 1;
		//	procState_ = TurnOffReinit;
		break;
		case TurnOffReinit:
			clock->doingReinit_ = 0;
			procState_ = NoChange;
		break;
		case ReinitThenStart:
			clock->doingReinit_ = 1;
			procState_ = StartOnly;
		break;
		case StartOnly:
			clock->doingReinit_ = 0;
			clock->isRunning_ = 1;
			procState_ = NoChange;
		break;
		case StopOnly:
			clock->isRunning_ = 0;
			procState_ = NoChange;
		break;
		case StopThenReinit:
			clock->isRunning_ = 0;
			clock->doingReinit_ = 1;
		//	procState_ = TurnOffReinit;
		break;
		case NoChange:
		default:
		break;
	}
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
 * Note that this is executed during the generic phase2 or phase3, in
 * parallel with lots of other threads. We cannot touch any fields that may
 * affect other threads.
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
	// isRunning_ = 1; // Can't touch this here, instead defer to barrier3
	if ( tickPtr_.size() == 0 || tickPtr_.size() != tickMgr_.size() || 
		!tickPtr_[0].mgr()->isInited() )
		procState_ = ReinitThenStart;
	else
		procState_ = StartOnly;
}

/// Static function
void Clock::reportClock() 
{
	const Clock* c = reinterpret_cast< const Clock* >( Id( 1 ).eref().data() );
	c->innerReportClock();
}

void Clock::innerReportClock() const
{
	cout << "reporting Clock: runTime= " << runTime_ << 
		", currentTime= " << currentTime_ << ", endTime= " << endTime_ <<
		", dt= " << dt_ << ", isRunning = " << isRunning_ << endl;
	cout << "uniqueDts= ";
	for ( unsigned int i = 0; i < tickPtr_.size(); ++i ) {
		cout << "  " << tickPtr_[i].mgr()->getDt() << "(" <<
		tickPtr_[i].mgr()->ticks().size() << ")";
	}
	cout << endl;
}

void Clock::handleStep( unsigned int numSteps )
{
	double runtime = dt_ * numSteps;
	handleStart( runtime );
}

// Advance system state by one clock tick. This may be a subset of
// one timestep, as there may be multiple clock ticks within one dt.
// This simply distributes the call to all scheduled objects
void Clock::advancePhase1(  ProcInfo *p )
{
	tickPtr_[0].mgr()->advancePhase1( p );
	if ( Shell::isSingleThreaded() || p->threadIndexInGroup == 1 ) {
		++countAdvance1_;
	}
}

// In phase 2 we need to do the updates to the Clock object, especially
// sorting the TickPtrs. This also is when we find out if the simulation
// is finished.
// Note that this function happens when lots of other threads are doing
// things. So it cannot touch any fields which might affect other threads.
void Clock::advancePhase2(  ProcInfo *p )
{
	if ( Shell::isSingleThreaded() || p->threadIndexInGroup == 1 ) {
		tickPtr_[0].mgr()->advancePhase2( p );
		if ( tickPtr_.size() > 1 )
			sort( tickPtr_.begin(), tickPtr_.end() );
		currentTime_ = tickPtr_[0].mgr()->getNextTime() - 
			tickPtr_[0].mgr()->getDt();
		if ( currentTime_ > endTime_ ) {
			Id clockId( 1 );
			procState_ = StopOnly;
			finished()->send( clockId.eref(), p->threadIndexInGroup );
			ack()->send( clockId.eref(), p->threadIndexInGroup, 
				p->nodeIndexInGroup, OkStatus );
		}
		++countAdvance2_;
	}
}

/////////////////////////////////////////////////////////////////////
// Scheduling the 'reinit' operation for scheduled objects.
// Three functions are involved: the handler for the reinit function,
// and the reinitPhase1 and reinitPhase2.
// Like the Process operation, reinit must go sequentially through all
// Tick Managers in order of increasing dt. Within each TickManager
// it must do the successive ticks in order of increasing index.
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
	currTickPtr_ = 0;

	// Get all the TickMgrs in increasing order of dt for the reinit call.
	for ( vector< TickMgr >::iterator i = tickMgr_.begin(); 
		i != tickMgr_.end(); ++i )
		i->reinitPhase0();
	if ( tickPtr_.size() > 1 )
		sort( tickPtr_.begin(), tickPtr_.end() );

	if ( isRunning_ )
		procState_ = StopThenReinit;
	else
		procState_ = TurnOnReinit;
	// flipReinit_ = 1; // This tells the clock to reinit in barrier3.
	// doingReinit_ = 1; // Can't do this here, may mess up other threads.
	// isRunning_ = 0; // Can't do this here either.
}


/**
 * Reinit is used to reinit the state of the scheduling system.
 * This version is meant to be done through the multithread scheduling
 * loop.
 * In phase1 it calls reinit on the target Elements.
 */
void Clock::reinitPhase1( ProcInfo* info )
{
	if ( tickPtr_.size() == 0 )
		return;
	assert( currTickPtr_ < tickPtr_.size() );
	tickPtr_[ currTickPtr_ ].mgr()->reinitPhase1( info );

	/*
	tickPtr_[0].mgr()->reinitPhase1( info );
	for ( vector< TickPtr >::const_iterator i = tickPtr_.begin();
		i != tickPtr_.end(); ++i ) {
		i->mgr()->reinitPhase1( info );
	}
	*/
	if ( Shell::isSingleThreaded() || info->threadIndexInGroup == 1 )
		++countReinit1_;
}

/**
 * In phase2 it advances the internal counter to move to the next tick,
 * and when all ticks for this TickManager are done, to move to the next
 * TickManager.
 */
void Clock::reinitPhase2( ProcInfo* info )
{
	info->currTime = 0.0;
	if ( Shell::isSingleThreaded() || info->threadIndexInGroup == 1 ) {
		assert( currTickPtr_ < tickPtr_.size() );
		if ( tickPtr_[ currTickPtr_ ].mgr()->reinitPhase2( info ) ) {
			++currTickPtr_;
			if ( currTickPtr_ >= tickPtr_.size() ) {
				Id clockId( 1 );
				ack()->send( clockId.eref(), info->threadIndexInGroup,
					info->nodeIndexInGroup, OkStatus );
				procState_ = TurnOffReinit;
				++countReinit2_;
			}
		}
	}
}

