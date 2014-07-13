/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2013 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

/**
 * The Clock manages simulation scheduling, in a close
 * collaboration with the Tick.
 * This version does this using an array of child Ticks, which
 * it manages directly.
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
 * This scheduler uses integral multiples of the base timestep dt_.
 *
 * The system works like this:
 * 1. Assign times to each Tick. This is divided by dt_ and the rounded
 * 		value is used for the integral multiple. Zero means the tick is not
 * 		scheduled.
 * 2. The process call goes through all active ticks in order every 
 * 		timestep. Each Tick increments its counter and decides if it is 
 * 		time to fire.
 * 4. The Reinit call goes through all active ticks in order, just once.
 * 5. We connect up the Ticks to their target objects.
 * 6. We begin the simulation by calling 'start' or 'step' on the Clock.
 * 		'step' executes exactly one timestep (of the minimum dt_), 
 * 		visiting all ticks as above..
 * 		'start' executes an integral number of such timesteps.
 * 7. To interrupt the simulation at some intermediate time, call 'stop'.
 * 		This lets the system complete its current step.
 * 8. To restart the simulation from where it left off, use the same 
 * 		'start' or 'step' function on the Clock. As all the ticks
 * 		retain their state, the simulation can resume smoothly.
 */

#include "header.h"
#include "Clock.h"

const unsigned int Clock::numTicks = 10;

///////////////////////////////////////////////////////
// MsgSrc definitions
///////////////////////////////////////////////////////

static SrcFinfo0 *finished() {
	static SrcFinfo0 finished( 
			"finished",
			"Signal for completion of run"
			);
	return &finished;
}

// This vector contains the SrcFinfos used for Process calls for each
// of the Ticks.
vector< SrcFinfo1< ProcPtr >* >& processVec() {
	static SrcFinfo1< ProcPtr > process0( "process0", "Process for Tick 0");
	static SrcFinfo1< ProcPtr > process1( "process1", "Process for Tick 1");
	static SrcFinfo1< ProcPtr > process2( "process2", "Process for Tick 2");
	static SrcFinfo1< ProcPtr > process3( "process3", "Process for Tick 3");
	static SrcFinfo1< ProcPtr > process4( "process4", "Process for Tick 4");
	static SrcFinfo1< ProcPtr > process5( "process5", "Process for Tick 5");
	static SrcFinfo1< ProcPtr > process6( "process6", "Process for Tick 6");
	static SrcFinfo1< ProcPtr > process7( "process7", "Process for Tick 7");
	static SrcFinfo1< ProcPtr > process8( "process8", "Process for Tick 8");
	static SrcFinfo1< ProcPtr > process9( "process9", "Process for Tick 9");
	static SrcFinfo1< ProcPtr > process10("process10", "Process Tick 10");
	static SrcFinfo1< ProcPtr > process11("process11", "Process Tick 11");
	static SrcFinfo1< ProcPtr > process12("process12", "Process Tick 12");
	static SrcFinfo1< ProcPtr > process13("process13", "Process Tick 13");
	static SrcFinfo1< ProcPtr > process14("process14", "Process Tick 14");
	static SrcFinfo1< ProcPtr > process15("process15", "Process Tick 15");
	static SrcFinfo1< ProcPtr > process16("process16", "Process Tick 16");
	static SrcFinfo1< ProcPtr > process17("process17", "Process Tick 17");
	static SrcFinfo1< ProcPtr > process18("process18", "Process Tick 18");
	static SrcFinfo1< ProcPtr > process19("process19", "Process Tick 19");

	static SrcFinfo1< ProcPtr >* processArray[] = {
		&process0, &process1, &process2, &process3, &process4, 
		&process5, &process6, &process7, &process8, &process9,
		&process10, &process11, &process12, &process13, &process14,
	   	&process15, &process16, &process17, &process18, &process19,
   	};
	static vector< SrcFinfo1< ProcPtr >* > processVec(processArray, processArray + sizeof(processArray) / sizeof(SrcFinfo1< ProcPtr > *));

	return processVec;
}

vector< SrcFinfo1< ProcPtr >* >& reinitVec() {

	static SrcFinfo1< ProcPtr > reinit0( "reinit0", "Reinit for Tick 0" );
	static SrcFinfo1< ProcPtr > reinit1( "reinit1", "Reinit for Tick 1" );
	static SrcFinfo1< ProcPtr > reinit2( "reinit2", "Reinit for Tick 2" );
	static SrcFinfo1< ProcPtr > reinit3( "reinit3", "Reinit for Tick 3" );
	static SrcFinfo1< ProcPtr > reinit4( "reinit4", "Reinit for Tick 4" );
	static SrcFinfo1< ProcPtr > reinit5( "reinit5", "Reinit for Tick 5" );
	static SrcFinfo1< ProcPtr > reinit6( "reinit6", "Reinit for Tick 6" );
	static SrcFinfo1< ProcPtr > reinit7( "reinit7", "Reinit for Tick 7" );
	static SrcFinfo1< ProcPtr > reinit8( "reinit8", "Reinit for Tick 8" );
	static SrcFinfo1< ProcPtr > reinit9( "reinit9", "Reinit for Tick 9" );
	static SrcFinfo1< ProcPtr > reinit10( "reinit10", "Reinit for Tick 10");
	static SrcFinfo1< ProcPtr > reinit11( "reinit11", "Reinit for Tick 11");
	static SrcFinfo1< ProcPtr > reinit12( "reinit12", "Reinit for Tick 12");
	static SrcFinfo1< ProcPtr > reinit13( "reinit13", "Reinit for Tick 13");
	static SrcFinfo1< ProcPtr > reinit14( "reinit14", "Reinit for Tick 14");
	static SrcFinfo1< ProcPtr > reinit15( "reinit15", "Reinit for Tick 15");
	static SrcFinfo1< ProcPtr > reinit16( "reinit16", "Reinit for Tick 16");
	static SrcFinfo1< ProcPtr > reinit17( "reinit17", "Reinit for Tick 17");
	static SrcFinfo1< ProcPtr > reinit18( "reinit18", "Reinit for Tick 18");
	static SrcFinfo1< ProcPtr > reinit19( "reinit19", "Reinit for Tick 19");

	static SrcFinfo1< ProcPtr >* reinitArray[] = {
		&reinit0, &reinit1, &reinit2, &reinit3, &reinit4, 
		&reinit5, &reinit6, &reinit7, &reinit8, &reinit9, 
		&reinit10, &reinit11, &reinit12, &reinit13, &reinit14, 
		&reinit15, &reinit16, &reinit17, &reinit18, &reinit19, 
	};
	static vector< SrcFinfo1< ProcPtr >* > reinitVec(reinitArray, reinitArray + sizeof(reinitArray) / sizeof(SrcFinfo1< ProcPtr > *));

	return reinitVec;
}

const Cinfo* Clock::initCinfo()
{
	///////////////////////////////////////////////////////
	// Field definitions
	///////////////////////////////////////////////////////
		static ValueFinfo< Clock, double > dt( 
			"dt",
			"Base timestep for simulation",
			&Clock::setDt,
			&Clock::getDt
		);
		static ReadOnlyValueFinfo< Clock, double > runTime( 
			"runTime",
			"Duration to run the simulation",
			&Clock::getRunTime
		);
		static ReadOnlyValueFinfo< Clock, double > currentTime(
			"currentTime",
			"Current simulation time",
			&Clock::getCurrentTime
		);
		static ReadOnlyValueFinfo< Clock, unsigned int > nsteps( 
			"nsteps",
			"Number of steps to advance the simulation, in units of the smallest timestep on the clock ticks",
			&Clock::getNsteps
		);
		static ReadOnlyValueFinfo< Clock, unsigned int > numTicks( 
			"numTicks",
			"Number of clock ticks",
			&Clock::getNumTicks
		);
		static ReadOnlyValueFinfo< Clock, unsigned int > currentStep( 
			"currentStep",
			"Current simulation step",
			&Clock::getCurrentStep
		);

		static ReadOnlyValueFinfo< Clock, vector< double > > dts( 
			"dts",
			"Utility function returning the dt (timestep) of all ticks.",
			&Clock::getDts
		);

		static ReadOnlyValueFinfo< Clock, bool > isRunning( 
			"isRunning",
			"Utility function to report if simulation is in progress.",
			&Clock::isRunning
		);

		static LookupValueFinfo< Clock, unsigned int, unsigned int > 
			tickStep(
			"tickStep",
			"Step size of specified Tick, as integral multiple of dt_"
			" A zero step size means that the Tick is inactive",
			&Clock::setTickStep,
			&Clock::getTickStep
		);

		static LookupValueFinfo< Clock, unsigned int, double > tickDt(
			"tickDt",
			"Timestep dt of specified Tick. Always integral multiple of "
			"dt_. If you assign a non-integer multiple it will round off. "
			" A zero timestep means that the Tick is inactive",
			&Clock::setTickDt,
			&Clock::getTickDt
		);
	///////////////////////////////////////////////////////
	// Shared definitions
	///////////////////////////////////////////////////////
	static Finfo* procShared0[] = { processVec()[0], reinitVec()[0] };
	static Finfo* procShared1[] = { processVec()[1], reinitVec()[1] };
	static Finfo* procShared2[] = { processVec()[2], reinitVec()[2] };
	static Finfo* procShared3[] = { processVec()[3], reinitVec()[3] };
	static Finfo* procShared4[] = { processVec()[4], reinitVec()[4] };
	static Finfo* procShared5[] = { processVec()[5], reinitVec()[5] };
	static Finfo* procShared6[] = { processVec()[6], reinitVec()[6] };
	static Finfo* procShared7[] = { processVec()[7], reinitVec()[7] };
	static Finfo* procShared8[] = { processVec()[8], reinitVec()[8] };
	static Finfo* procShared9[] = { processVec()[9], reinitVec()[9] };
	static Finfo* procShared10[] = { processVec()[10], reinitVec()[10] };
	static Finfo* procShared11[] = { processVec()[11], reinitVec()[11] };
	static Finfo* procShared12[] = { processVec()[12], reinitVec()[12] };
	static Finfo* procShared13[] = { processVec()[13], reinitVec()[13] };
	static Finfo* procShared14[] = { processVec()[14], reinitVec()[14] };
	static Finfo* procShared15[] = { processVec()[15], reinitVec()[15] };
	static Finfo* procShared16[] = { processVec()[16], reinitVec()[16] };
	static Finfo* procShared17[] = { processVec()[17], reinitVec()[17] };
	static Finfo* procShared18[] = { processVec()[18], reinitVec()[18] };
	static Finfo* procShared19[] = { processVec()[19], reinitVec()[19] };

	static string s = "Shared process/reinit message";
	unsigned int sz = sizeof( procShared0 ) / sizeof( const Finfo* );
	static SharedFinfo proc0( "proc0", s, procShared0, sz );
	static SharedFinfo proc1( "proc1", s, procShared1, sz );
	static SharedFinfo proc2( "proc2", s, procShared2, sz );
	static SharedFinfo proc3( "proc3", s, procShared3, sz );
	static SharedFinfo proc4( "proc4", s, procShared4, sz );
	static SharedFinfo proc5( "proc5", s, procShared5, sz );
	static SharedFinfo proc6( "proc6", s, procShared6, sz );
	static SharedFinfo proc7( "proc7", s, procShared7, sz );
	static SharedFinfo proc8( "proc8", s, procShared8, sz );
	static SharedFinfo proc9( "proc9", s, procShared9, sz );
	static SharedFinfo proc10( "proc10", s, procShared10, sz );
	static SharedFinfo proc11( "proc11", s, procShared11, sz );
	static SharedFinfo proc12( "proc12", s, procShared12, sz );
	static SharedFinfo proc13( "proc13", s, procShared13, sz );
	static SharedFinfo proc14( "proc14", s, procShared14, sz );
	static SharedFinfo proc15( "proc15", s, procShared15, sz );
	static SharedFinfo proc16( "proc16", s, procShared16, sz );
	static SharedFinfo proc17( "proc17", s, procShared17, sz );
	static SharedFinfo proc18( "proc18", s, procShared18, sz );
	static SharedFinfo proc19( "proc19", s, procShared19, sz );

	///////////////////////////////////////////////////////
	// MsgDest definitions
	///////////////////////////////////////////////////////
		static DestFinfo start( "start", 
			"Sets off the simulation for the specified duration",
			new EpFunc1< Clock, double >(&Clock::handleStart )
		);

		static DestFinfo step( "step", 
			"Sets off the simulation for the specified # of steps",
			new EpFunc1< Clock, unsigned int >(&Clock::handleStep )
		);

		static DestFinfo stop( "stop", 
			"Halts the simulation, with option to restart seamlessly",
			new OpFunc0< Clock >(&Clock::stop )
		);

		static DestFinfo reinit( "reinit", 
			"Zeroes out all ticks, starts at t = 0",
	 		new EpFunc0< Clock >(&Clock::handleReinit )
		);

		static Finfo* clockControlFinfos[] = {
			&start, &step, &stop, &reinit,
		};
	///////////////////////////////////////////////////////
	// SharedFinfo for Shell to control Clock
	///////////////////////////////////////////////////////
		static SharedFinfo clockControl( "clockControl",
			"Controls all scheduling aspects of Clock, usually from Shell",
			clockControlFinfos, 
			sizeof( clockControlFinfos ) / sizeof( Finfo* )
		);

	static Finfo* clockFinfos[] =
	{
		// Fields
		&dt,				// Value
		&runTime,			// ReadOnlyValue
		&currentTime,		// ReadOnlyValue
		&nsteps,			// ReadOnlyValue
		&numTicks,			// ReadOnlyValue
		&currentStep,		// ReadOnlyValue
		&dts,				// ReadOnlyValue
		&isRunning,			// ReadOnlyValue
		&tickStep,			// LookupValue
		&tickDt,			// LookupValue
		&clockControl,		// Shared
		finished(),			// Src
		&proc0,				// Src
		&proc1,				// Src
		&proc2,				// Src
		&proc3,				// Src
		&proc4,				// Src
		&proc5,				// Src
		&proc6,				// Src
		&proc7,				// Src
		&proc8,				// Src
		&proc9,				// Src
		&proc10,				// Src
		&proc11,				// Src
		&proc12,				// Src
		&proc13,				// Src
		&proc14,				// Src
		&proc15,				// Src
		&proc16,				// Src
		&proc17,				// Src
		&proc18,				// Src
		&proc19,				// Src
	};
	
	static string doc[] =
	{
		"Name", "Clock",
		"Author", "Upinder S. Bhalla, Nov 2013, NCBS",
		"Description", "Clock: Clock class. Handles sequencing of operations in simulations."
		"Every object scheduled for operations in MOOSE is connected to one"
		"of the 'Tick' entries on the Clock."
		"The Clock manages ten 'Ticks', each of which has its own dt,"
		"which is an integral multiple of the base clock dt_. "
		"On every clock step the ticks are examined to see which of them"
		"is due for updating. When a tick is updated, the 'process' call "
		"of all the objects scheduled on that tick is called."
		"The default scheduling (should not be overridden) has the "
		"following assignment of classes to Ticks:"
		"0. Biophysics: Init call on Compartments in EE method"
		"1. Biophysics: Channels"
		"2. Biophysics: Process call on Compartments"
		"3. Undefined "
		"4. Kinetics: Pools, or in ksolve mode: Mesh to handle diffusion"
		"5. Kinetics: Reacs, enzymes, etc, or in ksolve mode: Stoich/GSL"
		"6. Stimulus tables"
		"7. More stimulus tables"
		"8. Plots"
		"9. Postmaster. This must be called last of all and nothing else "
		"should use this Tick. The Postmaster is automatically scheduled "
		"at set up time. The Tick should be given the longest possible "
		"value, typically but not always equal to one of the other ticks, "
		"so as to batch the "
		"communications. For spiking-only communications, it is usually "
		"possible to space the communication tick by as much as 1-2 ms "
		"which is the axonal + synaptic delay. "
		"",
	};

	static Dinfo< Clock > dinfo;
	static Cinfo clockCinfo(
		"Clock",
		// "Clock class handles sequencing of operations in simulations",
		Neutral::initCinfo(),
		clockFinfos,
		sizeof(clockFinfos)/sizeof(Finfo *),
		&dinfo,
                doc,
                sizeof(doc)/sizeof(string)
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
	  nSteps_( 0 ), 
	  currentStep_( 0 ), 
	  dt_( 1.0 ), 
	  isRunning_( false ),
	  doingReinit_( false ),
	  info_(),
	  ticks_( Clock::numTicks, 0 )
{
}
///////////////////////////////////////////////////
// Field function definitions
///////////////////////////////////////////////////
void Clock::setDt( double v)
{
	if ( isRunning_ ) {
		cout << "Warning: Clock::setDt: Cannot change dt while simulation is running\n";
		return;
	}
	dt_ = v;
}

double Clock::getDt() const
{
	return dt_;
}
double Clock::getRunTime() const
{
	return runTime_;
}

double Clock::getCurrentTime() const
{
	return currentTime_;
}

unsigned int Clock::getNsteps() const
{
	return nSteps_;
}

unsigned int Clock::getCurrentStep() const
{
	return currentStep_;
}

unsigned int Clock::getNumTicks() const
{
	return numTicks;
}


vector< double > Clock::getDts() const
{
	vector< double > ret;
	for ( unsigned int i = 0; i < ticks_.size(); ++i ) {
		ret.push_back( ticks_[ i ] * dt_ );
	}
	return ret;
}

bool Clock::isRunning() const
{
	return isRunning_;
}

bool Clock::isDoingReinit() const
{
	return doingReinit_;
}

bool Clock::checkTickNum( const string& funcName, unsigned int i ) const
{
	if ( isRunning_ || doingReinit_) {
		cout << "Warning: Clock::" << funcName << 
				": Cannot change dt while simulation is running\n";
		return false;
	}
	if ( i >= Clock::numTicks ) {
		cout << "Warning: Clock::" << funcName <<
			"( " << i << " ): Clock has only " << 
			Clock::numTicks << " ticks \n";
		return false;
	}
	return true;
}

void Clock::setTickStep( unsigned int i, unsigned int v )
{
	if ( checkTickNum( "setTickStep", i ) )
		ticks_[i] = v;
}
unsigned int Clock::getTickStep( unsigned int i ) const
{
	if ( i < Clock::numTicks )
		return ticks_[i];
	return 0;
}

/**
 * A little nasty because we want to ensure that the main clock dt is
 * set intelligently from the assignment here.
 */
void Clock::setTickDt( unsigned int i, double v )
{
	unsigned int numUsed = 0;
	for ( unsigned int j = 0; j < numTicks; ++j )
		numUsed += ( ticks_[j] != 0 );
	if ( numUsed == 0 ) {
		dt_ = v;
	} else if ( dt_ > v ) {
		for ( unsigned int j = 0; j < numTicks; ++j )
			if ( ticks_[j] != 0 )
				ticks_[j] = round( ( ticks_[j] * dt_ ) / v );
		dt_ = v;
	}

	if ( checkTickNum( "setTickDt", i ) )
		ticks_[i] = round( v / dt_ );
}

double Clock::getTickDt( unsigned int i ) const
{
	if ( i < Clock::numTicks )
		return ticks_[i] * dt_;
	return 0.0;
}

///////////////////////////////////////////////////
// Dest function definitions
///////////////////////////////////////////////////

/**
 * Does a graceful stop of the simulation, leaving so it can continue
 * cleanly with another step or start command.
 */
void Clock::stop()
{
	isRunning_ = 0;
}

/////////////////////////////////////////////////////////////////////
// Info functions
/////////////////////////////////////////////////////////////////////

/// Static function
void Clock::reportClock() 
{
	const Clock* c = reinterpret_cast< const Clock* >( Id( 1 ).eref().data() );
	c->innerReportClock();
}

void Clock::innerReportClock() const
{
	cout << "reporting Clock: runTime= " << runTime_ << 
		", currentTime= " << currentTime_ <<
		", dt= " << dt_ << ", isRunning = " << isRunning_ << endl;
	cout << "Dts= ";
	for ( unsigned int i = 0; i < ticks_.size(); ++i ) {
		cout <<  "tick[" << i << "] = " << ticks_[i] << "	" <<
				ticks_[i] * dt_ << endl;
	}
	cout << endl;
}

/////////////////////////////////////////////////////////////////////
// Core scheduling functions.
/////////////////////////////////////////////////////////////////////

void Clock::buildTicks( const Eref& e )
{
	activeTicks_.resize(0);
	activeTicksMap_.resize(0);
	for ( unsigned int i = 0; i < ticks_.size(); ++i ) {
		if ( ticks_[i] > 0 && 
				e.element()->hasMsgs( processVec()[i]->getBindIndex() ) ) {
			activeTicks_.push_back( ticks_[i] );
			activeTicksMap_.push_back( i );
		}
	}
}

/**
 * Start has to happen gracefully: If the simulation was stopped for any
 * reason, it has to pick up where it left off.
 * The "runtime" argument is the additional time to run the simulation.
 */
void Clock::handleStart( const Eref& e, double runtime )
{
	unsigned int n = round( runtime / dt_ );
	handleStep( e, n );
}

void Clock::handleStep( const Eref& e, unsigned int numSteps )
{
	if ( isRunning_ || doingReinit_ ) {
		cout << "Clock::handleStart: Warning: simulation already in progress.\n Command ignored\n";
		return;
	}
	buildTicks( e );
	assert( currentStep_ == nSteps_ );
	assert( activeTicks_.size() == activeTicksMap_.size() );
	nSteps_ += numSteps;
	runTime_ = nSteps_ * dt_;
	for ( isRunning_ = true;
		isRunning_ && currentStep_ < nSteps_; ++currentStep_ )
	{
		// Curr time is end of current step.
		unsigned int endStep = currentStep_ + 1;
		currentTime_ = info_.currTime = dt_ * endStep;
		vector< unsigned int >::const_iterator k = activeTicksMap_.begin();
		for ( vector< unsigned int>::iterator j = 
			activeTicks_.begin(); j != activeTicks_.end(); ++j ) {
			if ( endStep % *j == 0 ) {
				info_.dt = *j * dt_;
				processVec()[*k]->send( e, &info_ );
			}
			++k;
		}
	}
	info_.dt = dt_;
	isRunning_ = false;
	finished()->send( e );
}

/**
 * This is the dest function that sets off the reinit.
 */
void Clock::handleReinit( const Eref& e )
{
	if ( isRunning_ || doingReinit_ ) {
		cout << "Clock::handleReinit: Warning: simulation already in progress.\n Command ignored\n";
		return;
	}
	currentTime_ = 0.0;
	currentStep_ = 0;
	nSteps_ = 0;
	buildTicks( e );
	doingReinit_ = true;
	// Curr time is end of current step.
	info_.currTime = 0.0;
	vector< unsigned int >::const_iterator k = activeTicksMap_.begin();
	for ( vector< unsigned int>::iterator j = 
		activeTicks_.begin(); j != activeTicks_.end(); ++j ) {
		info_.dt = *j * dt_;
		reinitVec()[*k++]->send( e, &info_ );
	}
	info_.dt = dt_;
	doingReinit_ = false;
}

/*
 * Useful function, only I don't need it yet. Was implemented for Dsolve
double Dsolve::findDt( const Eref& e )
{
	// Here is the horrible stuff to traverse the message to get the dt.
	const Finfo* f = Dsolve::initCinfo()->findFinfo( "reinit" );
	const DestFinfo* df = dynamic_cast< const DestFinfo* >( f );
	assert( df );
	unsigned int fid = df->getFid();
	ObjId caller = e.element()->findCaller( fid );
	const Msg* m = Msg::getMsg( caller );
	assert( m );
	vector< string > src = m->getSrcFieldsOnE1();
	assert( src.size() > 0 );
	string temp = src[0].substr( src[0].length() - 1 ); // reinitxx
	unsigned int tick = atoi( temp.c_str() );
	assert( tick < 10 );
	Id clock( 1 );
	assert( clock.element() == m->e1() );
	double dt = LookupField< unsigned int, double >::
			get( clock, "tickDt", tick );
	return dt;
}
*/
