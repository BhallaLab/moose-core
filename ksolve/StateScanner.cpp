/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2009 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

/**
 * This object orchestrates calculations using the gsl-solver and
 * the SteadyState object to do a number of related things:
 * - Make a dose-response curve
 * - try to find all steady states of a reaction
 * - Follow a state trajectory
 *
 * How to manage the various molecules etc:
 * 	1. Create an array of tables, one for each monitored molecule. 
 * 	Manipulate these tables using the Stack operations.
 * 	2. The tables themselves connect to the monitored molecules with an 
 * 	inputRequest message to keep track of values for each molecule: 
 * 	the points on the dose-response curve, or the steady states.
 * 	Control these through the Process message.
 *
 */

#include "moose.h"
#include "ProcInfo.h"
#include "../element/Neutral.h"
/*
#include "RateTerm.h"
#include "KinSparseMatrix.h"
#include "InterSolverFlux.h"
#include "Stoich.h"
*/
#include "StateScanner.h"

const double StateScanner::EPSILON = 1e-9;

const Cinfo* initStateScannerCinfo()
{
	/**
	 * This controls the stack operations of the x-axis child Table object
	 */
	static Finfo* stackShared[] =
	{
		new SrcFinfo( "push", Ftype1< double >::global() ),
		new SrcFinfo( "clear", Ftype0::global() ),
		new SrcFinfo( "pop", Ftype0::global() ),
	};

	/**
	 * This tells the other child Table objects to get hold of the
	 * latest mol conc from their attached molecules
	 */
	static Finfo* processShared[] =
	{
		new SrcFinfo( "process", Ftype1< ProcInfo >::global() ),
		new SrcFinfo( "reinit", Ftype1< ProcInfo >::global() ),
	};

	/**
	 * These are the fields of the StateScanner class
	 * It will have to spawn a number of tables to deal with the
	 * multiple solutions.
	 * Each table represents one molecule, and will have a separate
	 * entry for each solution. In the case of the dose-response, there
	 * will be a separate entry in each table for each of the dose values.
	 */
	static Finfo* stateScannerFinfos[] =
	{
		///////////////////////////////////////////////////////
		// Field definitions
		///////////////////////////////////////////////////////
		/*
		new ValueFinfo( "useTimeSeriesToSettle", 
			ValueFtype1< bool >::global(),
			GFCAST( &StateScanner::getSettleTime ), 
			RFCAST( &StateScanner::setSettleTime )
		),
		*/

		new ValueFinfo( "settleTime", 
			ValueFtype1< double >::global(),
			GFCAST( &StateScanner::getSettleTime ), 
			RFCAST( &StateScanner::setSettleTime )
		),
		new ValueFinfo( "solutionSeparation", 
			ValueFtype1< double >::global(),
			GFCAST( &StateScanner::getSolutionSeparation ), 
			RFCAST( &StateScanner::setSolutionSeparation ),
			"Threshold for RMS difference between solutions to classify them as distinct"
		),
		new LookupFinfo( "stateCategories",
			LookupFtype< unsigned int, unsigned int >::global(),
				GFCAST( &StateScanner::getStateCategory ),
				RFCAST( &StateScanner::setStateCategory ), // dummy
				"Look up categories obtained for each state"
		),
		new ValueFinfo( "useLog", 
			ValueFtype1< bool >::global(),
			GFCAST( &StateScanner::getUseLog ), 
			RFCAST( &StateScanner::setUseLog ),
			"Tells the scanner to use logarithmic sampling."
		),
		new ValueFinfo( "useSS", 
			ValueFtype1< bool >::global(),
			GFCAST( &StateScanner::getUseSS ), 
			RFCAST( &StateScanner::setUseSS ),
			"Tells the scanner to use the SteadyState solver for solutions."
		),
		new ValueFinfo( "useRisingDose", 
			ValueFtype1< bool >::global(),
			GFCAST( &StateScanner::getUseRisingDose ), 
			RFCAST( &StateScanner::setUseRisingDose ),
			"Tells the scanner to use rising levels of the dose (stimulus)."
		),
		new ValueFinfo( "useBufferDose", 
			ValueFtype1< bool >::global(),
			GFCAST( &StateScanner::getUseBufferDose ), 
			RFCAST( &StateScanner::setUseBufferDose ),
			"Tells the scanner to use buffered input (dose) concentrations."
		),
		new ValueFinfo( "classification",
			ValueFtype1< unsigned int >::global(),
			GFCAST( &StateScanner::getClassification ), 
			&dummyFunc,
			"Classification is derived for current system following a call"
			"to classifyStates. Outputs mean the following:"
			"0: Oscillatory"
			"1-7: # stable"
			"8: unclassified"
			"9: Oscillatory plus stable"
		),
		///////////////////////////////////////////////////////
		// MsgSrc definitions
		///////////////////////////////////////////////////////
		
		///////////////////////////////////////////////////////
		// MsgDest definitions
		///////////////////////////////////////////////////////
		new DestFinfo( "addTrackedMolecule", 
			Ftype1< Id >::global(),
			RFCAST( &StateScanner::addTrackedMolecule ),
			"addTrackedMolecule( molId ). "
		),
		new DestFinfo( "dropTrackedMolecule", 
			Ftype1< Id >::global(),
			RFCAST( &StateScanner::addTrackedMolecule ),
			"dropTrackedMolecule( molId ). "
		),
		
		new DestFinfo( "doseResponse", 
			Ftype4< Id, double, double, unsigned int>::global(),
			RFCAST( &StateScanner::doseResponse ),
			"doseResponse( molId, min, max, numSteps ). "
			"Do a dose response varying molId, from min to max, using numSteps"
		),
		new DestFinfo( "logDoseResponse", 
			Ftype4< Id, double, double, unsigned int>::global(),
			RFCAST( &StateScanner::logDoseResponse ),
			"logDoseResponse( molId, min, max, numSteps ). "
			"Do a dose response varying molId, from min to max,"
			"using numSteps in a logarithmic sequence."
		),
		new DestFinfo( "classifyStates", 
			Ftype3< unsigned int, bool, bool >::global(),
			RFCAST( &StateScanner::classifyStates ),
			"classifyStates( numStartingPoints, useMonteCarlo, useLog )"
			"Try to find and classify fixed points of this system, using"
			"settling from numStartingPoints initial states."
			"If the useMonteCarlo flag is zero, this "
			"is done using a systematic scan over the possible conc range."
			"If the useMonteCarlo flag is nonzero, this "
			"is done using MonteCarlo sampling"
			"If the useLog flag is true, it is done using logarithmic"
			"sampling over the possible conc range."
		),
		///////////////////////////////////////////////////////
		// Shared definitions
		///////////////////////////////////////////////////////
		/*
		new SharedFinfo( "gsl", gslShared, 
				sizeof( gslShared )/ sizeof( Finfo* ),
					"Messages that connect to the GslIntegrator object" ),

		new SharedFinfo( "stack", stackShared, 
			sizeof( stackShared )/ sizeof( Finfo* ),
			"Messages that connect to the Table objects, one per "
			"molecule, that handle arrays of results for each molecule." ),
		*/
		new SharedFinfo( "processSrc", processShared, 
			sizeof( processShared )/ sizeof( Finfo* ),
			"Messages that connect to the Table objects, one per "
			"molecule, that handle arrays of results for each molecule." ),
		new SharedFinfo( "stackSrc", stackShared, 
			sizeof( stackShared )/ sizeof( Finfo* ),
			"Messages to send x values for dose-response to another table"),
	};
	
	static string doc[] =
	{
		"Name", "StateScanner",
		"Author", "Upinder S. Bhalla, 2009, NCBS",
		"Description", "StateScanner: This object orchestrates "
		"calculations using the gsl-solver and the SteadyState object "
		"to do a number of related things: "
		"- Make a dose-response curve"
		"- try to find all steady states of a reaction"
		"- Follow a state trajectory"
		"When these operations are done, the system keeps track of all"
		"solutions for all specified molecules. It does so by creating"
		"a number of child tables."
	 	"Each table represents one molecule, and has a separate"
	    "entry for each solution. In the case of the dose-response, there"
	    "is a separate entry in each table for each of the dose values."
	};
	
	static Cinfo stateScannerCinfo(
		doc,
		sizeof( doc ) / sizeof( string ),		
		initNeutralCinfo(),
		stateScannerFinfos,
		sizeof( stateScannerFinfos )/sizeof(Finfo *),
		ValueFtype1< StateScanner >::global(),
		0,
		0
	);

	return &stateScannerCinfo;
}

static const Cinfo* stateScannerCinfo = initStateScannerCinfo();

static const Slot processSlot =
	initStateScannerCinfo()->getSlot( "processSrc.process" );

static const Slot reinitSlot =
	initStateScannerCinfo()->getSlot( "processSrc.reinit" );

static const Slot pushSlot =
	initStateScannerCinfo()->getSlot( "stackSrc.push" );
static const Slot clearSlot =
	initStateScannerCinfo()->getSlot( "stackSrc.clear" );
static const Slot popSlot =
	initStateScannerCinfo()->getSlot( "stackSrc.pop" );

///////////////////////////////////////////////////
// Class function definitions
///////////////////////////////////////////////////

StateScanner::StateScanner()
	:
		settleTime_( 1.0 ),
		solutionSeparation_( 1.0e-4),
		numTrackedMolecules_( 0 ),
		numSolutions_( 0 ),
		numStable_( 0 ),
		numSaddle_( 0 ),
		numOsc_( 0 ),
		numOther_( 0 ),
		classification_( 0 ),
		useLog_( 0 ),
		useRisingDose_( 1 ),
		useBufferDose_( 1 ),
		useSS_( 1 ),
		x_( 0.0 ),
		dx_( 0.0 ),
		lastx_( 0.0 ),
		end_( 0.0 )
{
	;
}

StateScanner::~StateScanner()
{
	;
}
		
///////////////////////////////////////////////////
// Field function definitions
///////////////////////////////////////////////////

unsigned int StateScanner::getClassification( Eref e ) {
	return static_cast< const StateScanner* >( e.data() )->classification_;
}

double StateScanner::getSettleTime( Eref e ) {
	return static_cast< const StateScanner* >( e.data() )->settleTime_;
}

void StateScanner::setSettleTime( const Conn* c, double value ) {
	if ( value >= 0.0 )
		static_cast< StateScanner* >( c->data() )->settleTime_ =
			value;
	else
		cout << "Warning: Settle Time " << value << 
		" must be positive. Old value: " << 
		static_cast< StateScanner* >( c->data() )->settleTime_ <<
		" retained\n";
}

double StateScanner::getSolutionSeparation( Eref e ) {
	return static_cast< const StateScanner* >( e.data() )->solutionSeparation_;
}

void StateScanner::setSolutionSeparation( const Conn* c, double value ) {
	if ( value > 0.0 )
		static_cast< StateScanner* >( c->data() )->solutionSeparation_ =
			value;
	else
		cout << "Warning: SolutionSeparation " << value << 
		" must be positive. Old value: " << 
		static_cast< StateScanner* >( c->data() )->solutionSeparation_ <<
		" retained\n";
}

bool StateScanner::getUseLog( Eref e ) {
	return static_cast< const StateScanner* >( e.data() )->useLog_;
}

void StateScanner::setUseLog( const Conn* c, bool value ) {
	static_cast< StateScanner* >( c->data() )->useLog_ = value;
}

bool StateScanner::getUseSS( Eref e ) {
	return static_cast< const StateScanner* >( e.data() )->useSS_;
}

void StateScanner::setUseSS( const Conn* c, bool value ) {
	static_cast< StateScanner* >( c->data() )->useSS_ = value;
}

bool StateScanner::getUseRisingDose( Eref e ) {
	return static_cast< const StateScanner* >( e.data() )->useRisingDose_;
}

void StateScanner::setUseRisingDose( const Conn* c, bool value ) {
	static_cast< StateScanner* >( c->data() )->useRisingDose_ = value;
}

bool StateScanner::getUseBufferDose( Eref e ) {
	return static_cast< const StateScanner* >( e.data() )->useBufferDose_;
}

void StateScanner::setUseBufferDose( const Conn* c, bool value ) {
	static_cast< StateScanner* >( c->data() )->useBufferDose_ = value;
}
		
///////////////////////////////////////////////////
// Utility function definitions
///////////////////////////////////////////////////

const string& uniqueName( Id elm )
{
	// Seketon function for now.
	return elm()->name();
}

bool isMolecule( Id elm )
{
	static const Cinfo* molCinfo = Cinfo::find( "Molecule" );
	if ( elm.good() ) {
		return ( elm()->cinfo()->isA( molCinfo ) );
		cout << "Warning: StateScanner: Element " << elm.path() << " is not a Molecule\n";
	}
	return 0;
}

///////////////////////////////////////////////////
// Dest function definitions
///////////////////////////////////////////////////

/** 
 * Creates a table and connects to it by a process message, and 
 * connects the table itself to the tracked molecule
 */
void StateScanner::addTrackedMolecule( const Conn* c, Id val )
{
	if ( !isMolecule( val ) )
		return;
	Eref scanner = c->target();
	string name = uniqueName( val );
	bool ret;
	Element* tab = Neutral::create( "Table", name, scanner.id(),
		Id::scratchId() );
	assert( tab->id().good() );
	Eref( tab ).dropAll( "process" ); // Eliminate the default msg.
	set< int >( Eref( tab ), "stepmode", 3 ); // TAB_BUF

	ret = Eref( tab ).add( "inputRequest", val(), "conc" );
	assert( ret );
	ret = scanner.add( "processSrc", tab, "process" );
	assert( ret );
}

/**
 * Finds the selected table and deletes it. The messages get automagically
 * cleared out.
 */
void StateScanner::dropTrackedMolecule( const Conn* c, Id val )
{
}


unsigned int StateScanner::getStateCategory( Eref e, const unsigned int& i )
{
	return static_cast< const StateScanner* >( e.data() )->localGetStateCategory(i);
}

void StateScanner::setStateCategory( 
	const Conn* c, unsigned int val, const unsigned int& i )
{
	; // dummy function.
}

unsigned int StateScanner::localGetStateCategory( unsigned int i ) const
{
	if ( i < stateCategories_.size() )
		return stateCategories_[i];
	cout << "Warning: StateScanner::localStateCategory: index " << i <<
			" out of range " << stateCategories_.size() << endl;
	return 0.0;
}


///////////////////////////////////////////////////
// Dest function definitions
///////////////////////////////////////////////////

// Static func
void StateScanner::doseResponse( const Conn* c,
	Id variableMol,
	double start, double end,
	unsigned int numSteps )
{
	static_cast< StateScanner* >( c->data() )->innerDoseResponse(
		c->target(), variableMol, start, end, numSteps, 0 );

}

void StateScanner::logDoseResponse( const Conn* c,
	Id variableMol,
	double start, double end,
	unsigned int numSteps )
{
	static_cast< StateScanner* >( c->data() )->innerDoseResponse(
		c->target(), variableMol, start, end, numSteps, 1 );
}

bool StateScanner::initDoser( double start, double end, unsigned int numSteps, bool useLog)
{
	if ( numSteps < 1 ) {
		cout << "Error: StateScanner::initDoser: numSteps < 1 \n";
		return 0;
	}
	if ( useLog_ ) {
		if ( start < EPSILON || end < EPSILON ) {
			cout << "Error: StateScanner:: initDoser: range of dose-response must not" << " go below zero in log mode\n";
			return 0;
		}
		dx_ = exp( log( end / start ) / numSteps );
	} else {
		dx_ = ( end - start ) / numSteps;
	}
	end_ = end;
	x_ = start;
	if ( start > end_ )
		useRisingDose_ = ( start > end_ );

	return 1;
}

bool StateScanner::advanceDoser()
{
	lastx_ = x_;
	if ( useLog_ ) {
		x_ *= dx_;
	} else {
		x_ += dx_;
	}
	if ( useRisingDose_ && x_ > end_ )
		return 0;
	if ( !useRisingDose_ && x_ < end_ )
		return 0;

	return 1;
}

void StateScanner::setControlParameter( Id& variableMol )
{
	if ( useBufferDose_ ) {
		set< double >( variableMol(), "concInit", x_ );
	} else {
		double conc;
		get< double >( variableMol(), "conc", conc );
		set< double >( variableMol(), "conc", conc + ( x_ - lastx_ ) );
	}
}

void StateScanner::settle( Eref me, Id& cj, Id& ss )
{
	static const Finfo* startFinfo = 
			Cinfo::find( "ClockJob" )->findFinfo( "start" );
	static const Finfo* settleFinfo = 
			Cinfo::find( "SteadyState" )->findFinfo( "settle" );
	static const Finfo* statusFinfo = 
			Cinfo::find( "SteadyState" )->findFinfo( "solutionStatus" );

	set< double >( cj(), startFinfo, settleTime_ );
	if ( useSS_ ) {
		set( ss(), settleFinfo );
		unsigned int status;
		get< unsigned int >( ss(), statusFinfo, status );
		if ( status != 0 ) { // Need to decide if to do fallback.
			// try running to steady state.
			cout << "status = 0\n";
		}
	}

	// Update the x axis (dose) array.
	send1< double >( me, pushSlot, x_ );
	// Update the molecule conc arrays.
	ProcInfoBase p( 0, settleTime_ );
	send1< ProcInfo >( me, processSlot, &p );
}

void StateScanner::makeDoseTable( Eref me )
{
	// Make the x axis table for the dose-response, if it isn't there.
	Id tab;
	lookupGet< Id, string >( me, "lookupChild", tab, "xAxis" );
	if ( tab == Id::badId() ) {
		tab = Id::scratchId();
		Neutral::create( "Interpol", "xAxis", me.id(), tab );
		assert( tab.good() );
		tab.eref().dropAll( "process" ); // Eliminate the default msg.
		bool ret = me.add( "stackSrc", tab.eref(), "stack" );
		assert( ret );
		if ( ret == 0 )
			cout << "StateScanner::makeDoseTable: Failed to add stack msg\n";
	}
}

/**
 * Dose-responses have the following options:
 * Linear vs. log
 * Using SS finder vs time-course
 * Specific buffered input molecule vs. total conc of mol
 * Reset vs incremented from previous solution
 * 		If continuing from previous solution: Rising vs falling
 *
 */
void StateScanner::innerDoseResponse( Eref me, Id variableMol,
	double start, double end,
	unsigned int numSteps,
	bool useLog)
{
	static const Finfo * concInitFinfo = 
		Cinfo::find( "Molecule" )->findFinfo( "concInit" );
	static const Finfo * modeFinfo = 
		Cinfo::find( "Molecule" )->findFinfo( "mode" );
	Id cj( "/sched/cj" );
	Id km( "/kinetics" );

	if ( !isMolecule( variableMol ) )
		return;
	makeDoseTable( me );
	double origConcInit = 0.0;
	int origMode = 0; // slave enable flag.
	get< double >( variableMol.eref(), concInitFinfo, origConcInit );
	get< int >( variableMol.eref(), modeFinfo, origMode );

	int newMode = ( useBufferDose_ ) ? 4 : 0;

	if ( newMode != origMode ) { // Need to rebuild the KineticManager
		set< int >( variableMol.eref(), modeFinfo, newMode );
		set< string >( km.eref(), "method", "ee" ); // Hack to force rebuild.
		set< string >( km.eref(), "method", "rk5" );
	}

	// Do a reset at the 'start' value
	set< double >( variableMol.eref(), concInitFinfo, start );
	set( cj(), "reinit" );

	// Pick up the correct entry in the 'totals' array of the SteadyState
	Id ss( "/kinetics/solve/ss" );

	if ( initDoser( start, end, numSteps, useLog ) == 0 )
		return;

	// Clear out the tables
	send0( me, clearSlot );
	ProcInfoBase p( 0, settleTime_ );
	send1< ProcInfo >( me, reinitSlot, &p );

	do {
		setControlParameter( variableMol );
		settle( me, cj, ss );
	} while ( advanceDoser() );

	// Figure out how to increment it.
	// Do the initial settling
	// In the loop: Assign the totals array. Do this first time too,
	// 	to get around numerical error in settling.
	// Go through loop, incrementing totals array and extracting state.
	// Do fallback time-series settle operation if the direct state
	// settling fails.

	set< double >( variableMol.eref(), concInitFinfo, origConcInit );

	if ( newMode != origMode ) { // Need to rebuild the KineticManager
		set< int >( variableMol.eref(), modeFinfo, origMode );
		set< string >( km.eref(), "method", "ee" ); // Hack to force rebuild.
		set< string >( km.eref(), "method", "rk5" );
	}
}

void StateScanner::classifyStates( const Conn* c,
	unsigned int numStartingPoints,
	bool useMonteCarlo,
	bool useLog )
{
	static_cast< StateScanner* >( c->data() )->innerClassifyStates(
		numStartingPoints,
		useMonteCarlo,
		useLog );
}

void StateScanner::innerClassifyStates(
		unsigned int numStartingPoints,
		bool useMonteCarlo,
		bool useLog )
{
}

///////////////////////////////////////////////////
// Other function definitions
///////////////////////////////////////////////////

/*
void StateScanner::assignStoichFunc( const Conn* c, void* s )
{
	static_cast< StateScanner* >( c->data() )->assignStoichFuncLocal( s );
}

void StateScanner::setMolN( const Conn* c, double y, unsigned int i )
{
}
*/

///////////////////////////////////////////////////
// GSL interface stuff
///////////////////////////////////////////////////

/**
 * This function should also set up the sizes, and it should be at 
 * allocate, not reinit time.
void StateScanner::assignStoichFuncLocal( void* stoich ) 
{
}
 */

