/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2005 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

/**
 * The ClockJob manages simulation scheduling, in a close
 * collaboration with the ClockTick.
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
 * This scheduler is quite general and handles any combination of
 * simulation times, including non-multiple ratios of dt.
 *
 * The ClockJob part of the team manages a set of ClockTicks.
 * Each ClockTick handles a given dt and a given stage within a dt.
 * The scheduler guarantees that the call sequence is preserved
 * between ClockTicks, but there are no sequence assumptions within
 * a single ClockTick.
 *
 * The system works like this:
 * 1. We create a bunch of ClockTicks on the ClockJob.
 * 2. We assign their dts and stages within each dt, if needed.
 * 3. We connect up the ClockTicks to their target objects.
 * 4. We call Resched on the ClockJob. 
 *    4.1 This strips off any pre-existing 'next' and 'prev' messages
 *    between the ticks (in case they had already been set up)
 *    4.2 It sorts the ClockTicks according to their dt and stage
 *    4.3 It sets up 'next' and 'prev' messages between them according
 *    to this sorted sequence.
 *    4.4 It sets up a 'next' and 'prev' message between itself and
 *    the first tick.
 *    4.5 On its local 'next' message to the first tick, it
 *    calls resched which trickles through all the ticks
 *    setting up state variables
 *    for their sequencing operations.
 * At this point we are all set up.
 * 5. We begin the simulation by calling 'start' or 'step' on the
 * ClockJob.
 *    5.1. Either of these functions work out the runTime and use it
 *    to call 'start' on the local 'next' message to the first tick.
 *    5.2 The start function on the first tick is the main loop for
 *    advancing time. It does the following:
 *       5.2.1 Sets up state variables and the dt in the ProcInfo
 *       Sets up endTime to either the end of the runtime (which
 *       is fine if there are no other ticks) or to the time that
 *       the next clock is due to fire, at nextClockTime.
 *       5.2.2 Loops till the current tick passes endTime, sending
 *       out process calls to all targets of current tick.
 *       5.2.3 Checks if there are any other ticks. If so (as usually
 *       the case), sends out the 'next' message to them with the
 *       time that they have to advance past.
 *           5.2.3.1. This 'next' message invokes the 'incrementClock'
 *           function on the next tick.
 *           5.2.3.2. In the incrementClock function, the next tick
 *           first sends out process calls.
 *           5.2.3.3. If there are any further ticks, it calls its
 *           own 'next' message to the further incrementClock function.
 *           5.2.3.4 Finally, it calls back to the originating 
 *           object telling it what its new value of nextTime_ is.
 *       5.2.4. Repeats this 'next' message till info comes back
 *       from the next tick to update the nextClockTime past the
 *       nextTime_ value.
 *    5.3 When the start function finally returns, the simulation is
 *    done till the desired time. The ClockJob updates its currentTime_
 *    field and returns.
 *
 * 6. We may wish to interrupt the simulation at some intermediate time
 * In principle this should be easy to do by putting in a 'halt'
 * flag. But it isn't implemented yet.
 * 7. We may wish to restart the simulation from where it left off.
 * This is done using the same 'start' or 'step' function on the
 * ClockJob. As all the ticks retain their state, the simulation can
 * resume smoothly.
 */

#include "moose.h"
#include "../element/Neutral.h"
#include "ClockJob.h"
#include "../shell/Shell.h"

const int ClockJob::emptyCallback = 0;
const int ClockJob::doReschedCallback = 1;

const Cinfo* initClockJobCinfo()
{
	static Finfo* tickShared[] = 
	{
		new SrcFinfo( "incrementTick", 
			Ftype2< ProcInfo, double >::global(),
			"This first entry is for the incrementTick function" ),
		new SrcFinfo( "nextTime", Ftype0::global(),
			"The second entry is a request to send nextTime_ from the next tick to the current one. " ),
		new DestFinfo( "recvNextTime", Ftype1< double >::global(), 
			RFCAST( &ClockJob::receiveNextTime ),
			"The third entry is for receiving the nextTime_ value from the following tick." ),
		new SrcFinfo( "resched", Ftype0::global(),
			"propagating resched forward." ),
		new SrcFinfo( "reinit", Ftype1< ProcInfo >::global(),
			"propagating reinit forward." ),
		new SrcFinfo( "reinitClock", Ftype0::global(),
			"propagating reinit forward." ),
		new SrcFinfo( "stopSrc", Ftype1< int >::global(),
			"invoke clean termination with a callback flag" ),
		new DestFinfo( "stopCallback", Ftype1< int >::global(),
			RFCAST( &ClockJob::handleStopCallback ),
			"callback from termination" ),
		new SrcFinfo( "checkRunning", Ftype0::global() ),
		new DestFinfo( "runningCallback", Ftype1< bool >::global(),
			RFCAST( &ClockJob::handleRunningCallback ),
			"callback from termination" ),
	};

	static Finfo* clockFinfos[] =
	{
	///////////////////////////////////////////////////////
	// Field definitions
	///////////////////////////////////////////////////////
		new ValueFinfo( "runTime", ValueFtype1< double >::global(),
			GFCAST( &ClockJob::getRunTime ), 
			RFCAST( &ClockJob::setRunTime )
			),
		new ValueFinfo( "currentTime", ValueFtype1< double >::global(),
			GFCAST( &ClockJob::getCurrentTime ), 
			&dummyFunc
			),
		new ValueFinfo( "nsteps", ValueFtype1< int >::global(),
			GFCAST( &ClockJob::getNsteps ), 
			RFCAST( &ClockJob::setNsteps )
			),
		new ValueFinfo( "currentStep", ValueFtype1< int >::global(),
			GFCAST( &ClockJob::getCurrentStep ), 
			&dummyFunc
			),
                new ValueFinfo( "autoschedule", ValueFtype1< int > ::global(),
                                GFCAST( & ClockJob::getAutoschedule),
                                RFCAST( & ClockJob::setAutoschedule),
                                "If 0, disable autoscheduling. Autoscheduling is enabled by default."),
	///////////////////////////////////////////////////////
	// Shared definitions
	///////////////////////////////////////////////////////
		// Connects up to the 'prev' shared message of the first
		// Tick in the sequence.
		new SharedFinfo( "tick", tickShared, 
			sizeof( tickShared ) / sizeof( Finfo* ),
			"This is a shared message that connects up to the 'prev' message of the first Tick in the "
			"sequence.It is equivalent to the Tick::next shared message. It invokes its incrementTick  "
			"function and also manages various functions for reset and return values. It is meant to  "
			"handle only a  single target." ),
	///////////////////////////////////////////////////////
	// MsgSrc definitions
	///////////////////////////////////////////////////////
		// A trigger when the simulation ends
		new SrcFinfo( "finishedSrc", Ftype0::global() ),

		// Sends ProcInfo and the runtime to the first Tick
		new SrcFinfo( "startSrc", Ftype2< ProcInfo, double >::global()),

	///////////////////////////////////////////////////////
	// MsgDest definitions
	///////////////////////////////////////////////////////
		new DestFinfo( "start", Ftype1< double >::global(),
					RFCAST( &ClockJob::startFunc ),
					"This function sets off the simulation for the specified duration" ),
		new DestFinfo( "step", Ftype1< int >::global(),
					RFCAST( &ClockJob::stepFunc ),
					"This sets of the simulation for the specified # of steps" ),
		new DestFinfo( "reinitClock", Ftype0::global(),
					RFCAST( &ClockJob::reinitClockFunc ) ),
		new DestFinfo( "reinit", Ftype0::global(),
					RFCAST( &ClockJob::reinitFunc ) ),
		new DestFinfo( "resched", Ftype0::global(),
					RFCAST( &ClockJob::reschedFunc ) ),

	///////////////////////////////////////////////////////
	// Synapse definitions
	///////////////////////////////////////////////////////
	};
	
	static string doc[] =
	{
		"Name", "ClockJob",
		"Author", "Upinder S. Bhalla, Mar 2007, NCBS",
		"Description", "ClockJob: ClockJob class. Handles sequencing of operations in simulations",
	};

	static Cinfo clockJobCinfo(
		doc,
		sizeof( doc ) / sizeof( string ),
		initNeutralCinfo(),
		clockFinfos,
		sizeof(clockFinfos)/sizeof(Finfo *),
		ValueFtype1< ClockJob >::global()
	);

	return &clockJobCinfo;
}

static const Cinfo* clockJobCinfo = initClockJobCinfo();

static const Slot startSlot = 
	initClockJobCinfo()->getSlot( "startSrc" );
static const Slot incrementSlot = 
	initClockJobCinfo()->getSlot( "tick.incrementTick" );
static const Slot requestNextTimeSlot = 
	initClockJobCinfo()->getSlot( "tick.nextTime" );
static const Slot reschedSlot = 
	initClockJobCinfo()->getSlot( "tick.resched" );
static const Slot reinitSlot = 
	initClockJobCinfo()->getSlot( "tick.reinit" );
static const Slot reinitClockSlot = 
	initClockJobCinfo()->getSlot( "tick.reinitClock" );
static const Slot stopSlot = 
	initClockJobCinfo()->getSlot( "tick.stopSrc" );
static const Slot checkRunningSlot = 
	initClockJobCinfo()->getSlot( "tick.checkRunning" );

///////////////////////////////////////////////////
// Constructor
///////////////////////////////////////////////////
ClockJob::ClockJob()
	: runTime_( 0.0 ), 
	  currentTime_( 0.0 ), 
	  nextTime_( 0.0 ),
	  nSteps_( 0 ), 
	  currentStep_( 0 ), 
	  dt_( 1.0 ), 
	  isRunning_( 0 ),
	  info_(),
	  callback_( emptyCallback )
{;}
///////////////////////////////////////////////////
// Field function definitions
///////////////////////////////////////////////////

/**
 * This sets the runtime to use for the simulation.
 * Perhaps this should not be assignable, but readonly.
 * Or The assignment should be a message.
 */
void ClockJob::setRunTime( const Conn* c, double v )
{
	static_cast< ClockJob* >( c->data() )->runTime_ = v;
}
double ClockJob::getRunTime( Eref e )
{
	return static_cast< ClockJob* >( e.data() )->runTime_;
}

double ClockJob::getCurrentTime( Eref e )
{
	return static_cast< ClockJob* >( e.data() )->info_.currTime_;
}

/**
 * This sets the number of steps to use for the simulation.
 * Perhaps this should not be assignable, but readonly.
 * Or The assignment should be a message.
 * It is a variant of the runtime function.
 */
void ClockJob::setNsteps( const Conn* c, int v )
{
	static_cast< ClockJob* >( c->data() )->nSteps_ = v;
}
int ClockJob::getNsteps( Eref e )
{
	return static_cast< ClockJob* >( e.data() )->nSteps_;
}

int ClockJob::getCurrentStep( Eref e )
{
	return static_cast< ClockJob* >( e.data() )->currentStep_;
}

int ClockJob::getAutoschedule( Eref e )
{
    return static_cast< ClockJob* >( e.data() )->autoschedule_;
}

void ClockJob::setAutoschedule( const Conn* c, int value )
{
    static_cast< ClockJob* >( c->data() )->autoschedule_ = value;
}

///////////////////////////////////////////////////
// Dest function definitions
///////////////////////////////////////////////////

void ClockJob::receiveNextTime( const Conn* c, double time )
{
	static_cast< ClockJob* >( c->data() )->nextTime_ = time;
}

void ClockJob::startFunc( const Conn* c, double runtime)
{
	static_cast< ClockJob* >( c->data() )->startFuncLocal( 
					c->target(), runtime );
}

void ClockJob::startFuncLocal( Eref e, double runTime )
{
	// cout << "starting run on " << Shell::myNode() << " for " << runTime << " sec.\n";

	isRunning_ = 1;
	send2< ProcInfo, double >( e, startSlot, &info_, 
					info_.currTime_ + runTime );
	isRunning_ = 0;
	/*
	info_.currTime_ = currentTime_;
	if ( tick_ )
		tick_->start( &info_, currentTime_ + runTime,
			processSrc_ );
	currentTime_ = info_.currTime_;
	*/
	// Finally we are out of all the loops. Do the resched if needed.
	if ( callback_ == doReschedCallback )
		reschedFuncLocal( e );
	callback_ = emptyCallback;
}

void ClockJob::stepFunc( const Conn* c, int nsteps )
{
	ClockJob* cj = static_cast< ClockJob* >( c->data() );
	cj->startFuncLocal( c->target(), nsteps * cj->dt_ );
}

/**
 * ReinitClock is used to reinit the state of the scheduling system.
 * This does not send out reinit calls to objects connected to ticks.
 */
void ClockJob::reinitClockFunc( const Conn* c )
{
	static_cast< ClockJob* >( c->data() )->
		reinitClockFuncLocal( c->target() );
}

void ClockJob::reinitClockFuncLocal( Eref e )
{
	info_.currTime_ = 0.0;
	runTime_ = 0.0;
	currentTime_ = 0.0;
	nextTime_ = 0.0;
	nSteps_ = 0;
	currentStep_ = 0;
	send0( e, reinitClockSlot );
}

/**
 * This function resets the schedule to zero time. It does NOT
 * reorder any of the clock ticks, it assumes that they are scheduled
 * correctly
 */
void ClockJob::reinitFunc( const Conn* c )
{
	static_cast< ClockJob* >( c->data() )->reinitFuncLocal(
					c->target() );
}
void ClockJob::reinitFuncLocal( Eref e )
{
	currentTime_ = 0.0;
	nextTime_ = 0.0;
	currentStep_ = 0;
	send1< ProcInfo >( e, reinitSlot, &info_ );
}

/**
 * The resched function reorders all clock ticks according to their
 * current dts. This involves clearing out their internal next and
 * prev messages (which effectively set them up in a linked list),
 * and then rebuilding this list according to the new sorted order.
 * This function does NOT mess with the current simulation time: you
 * can resched an ongoing simulation.
 */
void ClockJob::reschedFunc( const Conn* c )
{
	ClockJob* cj = static_cast< ClockJob* >( c->data() );
	// cj->isRunning_ = 0;
	// send0( c->target(), checkRunningSlot );
	if ( cj->isRunning_ ) {
		send1< int >( c->target(), stopSlot, doReschedCallback );
	} else {
		static_cast< ClockJob* >( c->data() )->reschedFuncLocal( 
			c->target() );
	}

	/*
	static const int tickMsg = clockJobCinfo->findFinfo( "tick" )->msg();
	if ( c->target()->numTargets( tickMsg ) > 0 )
		send1< int >( c->target(), stopSlot, doReschedCallback );
	else 
		static_cast< ClockJob* >( c->data() )->reschedFuncLocal( 
			c->target() );
	*/
}

class TickSeq {
	public:
			TickSeq()
			{;}

			TickSeq( Id id)
					: e_( id.eref() )
			{
					get< double >( e_, "dt", dt_ );
					get< int >( e_, "stage", stage_ );
			}

			bool operator<( const TickSeq& other ) const {
				if ( dt_ < other.dt_ ) return 1;
				if ( dt_ == other.dt_ && stage_ < other.stage_ )
						return 1;
				return 0;
			}

			Eref element() {
					return e_;
			}

	private:
			Eref e_;
			double dt_;
			int stage_;
};

/**
 * Builds up and sorts a list of clock Ticks. Ignores ones which have
 * no targets.
 */
void ClockJob::reschedFuncLocal( Eref er )
{
	vector< Id > childList = Neutral::getChildList( er.e );
	if ( childList.size() == 0 )
			return;
        // Commented this to avoid remapping of solvers to clock ticks
        // need to check how it affects the simulator
        // check the discussion on autoscheduling design

	vector< TickSeq > tickList;

	vector< TickSeq >::iterator j;
	vector< Id >::iterator i;

	// Clear out old tick list
	for ( i = childList.begin(); i != childList.end(); i++ ) {
		tickList.push_back( TickSeq( *i ) );
		for ( j = tickList.begin(); j != tickList.end(); j++ )
			clearMessages( j->element() );
	}
	tickList.resize( 0 );

	unsigned int totTargets = 0;
	for ( i = childList.begin(); i != childList.end(); i++ ) {
		const Finfo* procFinfo = (*i)()->findFinfo( "process" );
		assert ( procFinfo != 0 );
		unsigned int numTargets = ( *i )()->numTargets( procFinfo->msg() );
		// numTargets += ( *i )()->numTargets( "parTick" );

		procFinfo = (*i)()->findFinfo( "outgoingProcess" );
		if ( procFinfo != 0 )
			numTargets += ( *i )()->numTargets( procFinfo->msg() );
		// numTargets += ( *i )()->numTargets( "parTick" );

		// if ( numTargets > 0 )
			tickList.push_back( TickSeq( *i ) );
		totTargets += numTargets;
	}

	if ( tickList.size() == 0 )
		return;

	sort( tickList.begin(), tickList.end() );
	// cout << "Sorted ticklist with " << tickList.size() << " ticks and " << totTargets << " targets on node " << Shell::myNode() << endl;

	Eref last = tickList.front().element();
	bool ret = er.add( "tick", last, "prev" );
	assert( ret );
	ret = er.add( "startSrc", last, "start" );
	assert( ret );
	// cout << "Ticklist: " << last->name() << "	";
	for ( j = tickList.begin() + 1; j != tickList.end(); j++ ) {
		bool ret = last.add( "next", j->element(), "prev" );
		assert( ret );
			// buildMessages( last, j->element() );
		last = j->element();
		// cout << last->name() << "	";
	}
	// cout << endl << flush;
	send0( er, reschedSlot );
}

void ClockJob::handleStopCallback( const Conn* c, int flag )
{
	ClockJob* cj = static_cast< ClockJob* >( c->data() );
	cout << "Handling stop callback = " << flag << " on node " << Shell::myNode() << " at time = " << cj->info_.currTime_ << endl << flush;
	cj->callback_ = flag;
}

void ClockJob::handleRunningCallback( const Conn* c, bool flag )
{
	static_cast< ClockJob* >( c->data() )->isRunning_ = flag;
}


/**
 * ClearMessages removes the next/prev messages from each Tick.
 * It does not touch the process messages from the Tick to the 
 * objects it controls.
 */
void ClockJob::clearMessages( Eref e )
{
	e.dropAll( "prev" );
	e.dropAll( "start" );
}

/**
 * BuildMessages sets up the next/prev messages for each Tick.
 * deprecated
void ClockJob::buildMessages( Element* last, Element* e )
{
	bool ret = 
		last->findFinfo( "next" )->add( last, e, e->findFinfo( "prev" ) );
	assert( ret );
}
 */


/**
 * This function handles any changes to dt in the ticks. This means
 * it must redo the ordering of the ticks and call a resched on them.
 * \todo Currently only a placeholder.
 */
void ClockJob::dtFunc( const Conn* c, double dt )
{
	static_cast< ClockJob* >( c->data() )->dtFuncLocal( c->target(), dt );
}

void ClockJob::dtFuncLocal( Eref e, double dt )
{
		/*
	if ( tick_ == 0 )
		return;
	unsigned long index = clockConn_.find( tick ); 
	tick_->updateDt( dt, index ); 
	sortTicks(); 
	*/
}

///////////////////////////////////////////////////
// Other function definitions
///////////////////////////////////////////////////

/*
void ClockJob::sortTicks( )
{
	ClockTickMsgSrc** i;
	ClockTickMsgSrc** j = &tick_;
	bool swapping = 1;
	while ( swapping ) {
		swapping = 0;
		for ( i = (*j)->next(); *i != 0; i = (*i)->next() ) {
			if ( **i < **j ) {
				(*i)->swap( j );
				i = j;
				swapping = 1;
				break;
			} else {
				j = i;
			}
		}
	}
	tick_->updateNextClockTime();
}
*/
