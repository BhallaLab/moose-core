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

#include <fstream>
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
			Ftype2< unsigned int, bool >::global(),
			RFCAST( &StateScanner::classifyStates ),
			"classifyStates( numStartingPoints, useMonteCarlo )"
			"Try to find and classify fixed points of this system, using"
			"settling from numStartingPoints initial states."
			"If the useMonteCarlo flag is zero, this "
			"is done using a systematic scan over the possible conc range."
			"If the useMonteCarlo flag is nonzero, this "
			"is done using MonteCarlo sampling"
		),
		new DestFinfo( "saveAsXplot", 
			Ftype1< string >::global(),
			RFCAST( &StateScanner::saveAsXplot ),
			"Save as an xplot file, with successive plots separated by /newplot and /plotname <name>."
			"This format has two columns. Left column has xAxis value, right column has data."
			"The xAxis values are repeated for each plot."
		),
		new DestFinfo( "saveAsCSV", 
			Ftype1< string >::global(),
			RFCAST( &StateScanner::saveAsCSV ),
			"Save as a single flat CSV (comma separated value) table, with"
			"the left column having the xAxis values, and successive plots"
			"following it in later columns."
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
		useBufferDose_( 0 ),
		useReinit_( 1 ),
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
// Utility function definitions
///////////////////////////////////////////////////

const string uniqueName( Id elm )
{
	// Seketon function for now.
	Id pa = Neutral::getParent( elm.eref() );
	string name = pa()->name() + "_" + elm()->name();
	return name;
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
// Simple Dest function definitions
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

/**
 * Looks up all table solutions and organizes data into a 2d array
 * The first entry is the state or xAxis.
 * The remaining entries are the molecule-specific tables.
 */
unsigned int getTables( Eref me, vector< string >& names,
	vector< vector< double > >& solutions )
{
	const Finfo* tabFinfo = Cinfo::find( "Interpol" )->findFinfo( "tableVector" );

	Conn* c = me->targets( "stackSrc", me.i );
	if ( c->good() ) {
		names.push_back( c->target()->name() );
		vector< double > temp;
		get< vector< double > >( c->target(), tabFinfo, temp );
		solutions.push_back( temp );
	}
	delete c;
	c = me->targets( "processSrc", me.i );
	// Get vectors for each child table
	while ( c->good() ) {
		names.push_back( c->target()->name() );
		vector< double > temp;
		get< vector< double > >( c->target(), tabFinfo, temp );
		solutions.push_back( temp );
		c->increment();
	}
	assert( solutions.size() == names.size() );
	delete c;
	if ( solutions.size() == 0 )
		cout << "Warning: StateScanner::getTables: No solutions found\n";
	return solutions.size();
}


/**
 * Saves output as a CSV file. xAxis table entries are in leftmost column.
 */
void StateScanner::saveAsCSV( const Conn* conn, string fname )
{
	ofstream fout( fname.c_str(), std::ios::trunc );

	vector< string > names;
	vector< vector< double > > solutions;
	unsigned int n = getTables( conn->target(), names, solutions );
	if ( n == 0 )
		return;

	fout << "Solution_number,";
	for ( unsigned int i = 0; i < n - 1; ++i )
		fout << names[i] << ",";
	fout << names[ n-1 ] << endl;		

	for ( unsigned int i = 0; i < solutions[0].size(); ++i ) {
		fout << i << ",";
		for ( unsigned int j = 0; j < n - 1; ++j )
			fout << solutions[j][i] << ",";
		fout << solutions[ n-1 ][i] << endl;		
	}
}

/**
 * Saves output as an xplot file. xAxis table entries are in left column.
 */
void StateScanner::saveAsXplot( const Conn* c, string fname )
{
	ofstream fout( fname.c_str(), std::ios::trunc );
	vector< string > names;
	vector< vector< double > > solutions;
	unsigned int n = getTables( c->target(), names, solutions );
	if ( n == 0 )
		return;
	unsigned int size = solutions[0].size();

	for ( unsigned int i = 1; i < n; ++i ) {
		fout << "/newplot\n";
		fout << "/plotname " << names[ i ] << "\n";
		for ( unsigned int j = 0; j < size; ++j )
			fout << solutions[0][j] << "	" << solutions[i][j] << endl;
		fout << endl;
	}
}

///////////////////////////////////////////////////
// Complex Dest function definitions
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


void StateScanner::classifyStates( const Conn* c,
	unsigned int numStartingPoints, bool useMonteCarlo )
{
	static_cast< StateScanner* >( c->data() )->innerClassifyStates(
		c->target(), numStartingPoints, useMonteCarlo );
}

///////////////////////////////////////////////////
// Helper function definitions for dose response
///////////////////////////////////////////////////

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
	static const Finfo* reinitFinfo = 
			Cinfo::find( "ClockJob" )->findFinfo( "reinit" );
	static const Finfo* settleFinfo = 
			Cinfo::find( "SteadyState" )->findFinfo( "settle" );
	static const Finfo* statusFinfo = 
			Cinfo::find( "SteadyState" )->findFinfo( "solutionStatus" );

	if ( useReinit_ && useBufferDose_ )
		set( cj(), reinitFinfo );
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

void StateScanner::makeChildTable( Eref me, string name )
{
	// Make the x axis table for the dose-response, if it isn't there.
	Id tab;
	lookupGet< Id, string >( me, "lookupChild", tab, name );
	if ( tab == Id::badId() ) {
		tab = Id::scratchId();
		Neutral::create( "Interpol", name, me.id(), tab );
		assert( tab.good() );
		tab.eref().dropAll( "process" ); // Eliminate the default msg.
		bool ret = me.add( "stackSrc", tab.eref(), "stack" );
		assert( ret );
		if ( ret == 0 )
			cout << "StateScanner::makeChildTable: Failed to add stack msg\n";
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
	makeChildTable( me, "xAxis" );
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


	set< double >( variableMol.eref(), concInitFinfo, origConcInit );

	if ( newMode != origMode ) { // Need to rebuild the KineticManager
		set< int >( variableMol.eref(), modeFinfo, origMode );
		set< string >( km.eref(), "method", "ee" ); // Hack to force rebuild.
		set< string >( km.eref(), "method", "rk5" );
	}
}

////////////////////////////////////////////////////////////////
// Helper function definitions for state classification
////////////////////////////////////////////////////////////////

/**
 * In order to control the scanning through initial conditions, we
 * need to know all about the gamma (conservation) matrix and the 
 * vector of totals (which is not necessarily the simple sum of mol
 * concs). Rather than do that here, we need to talk to the SteadyState
 * object and ask it to do the scanning, as it knows all.
 *
 * Should add option to pick its own molecules to monitor.
 */
void StateScanner::innerClassifyStates(
		Eref me,
		unsigned int numStartingPoints,
		bool useMonteCarlo
		)
{
	static const Finfo* randomInitFinfo = 
			Cinfo::find( "SteadyState" )->findFinfo( "randomInit" );

	Id cj( "/sched/cj" );
	Id km( "/kinetics" );
	Id ss( "/kinetics/solve/ss" );

	makeChildTable( me, "state" );
	set( cj(), "reinit" );
	// Pick up the correct entry in the 'totals' array of the SteadyState

	// Clear out the tables
	send0( me, clearSlot );
	ProcInfoBase p( 0, settleTime_ );
	send1< ProcInfo >( me, reinitSlot, &p );

	for ( unsigned int i = 0; i < numStartingPoints; ++i ) {
		set( ss(), randomInitFinfo );
		if ( stateSettle( me, cj, ss ) )
			checkIfUniqueState( me );
	}
	classify();
}



void StateScanner::classify() 
{
	// Get the values in the state vector
	// Think about them.
	Id stateId( "/kinetics/scan/state" );

	vector< double > state;
	get< vector< double > >( stateId.eref(), "tableVector", state );
	unsigned int numSolutions = state.size();
	unsigned int numStable = 0;
	unsigned int numUnstable = 0;
	unsigned int numSaddle = 0;
	unsigned int numOsc = 0;
	unsigned int numSingleZero = 0;
	unsigned int numOther = 0;
	for ( unsigned int i = 0; i < numSolutions; ++i ) {
		switch ( int( state[i] + EPSILON ) ) {
			case 0: ++numStable;
				break;
			case 1: ++numUnstable;
				break;
			case 2: ++numSaddle;
				break;
			case 3: ++numOsc;
				break;
			case 4: ++numSingleZero;
				break;
			case 5: ++numOther;
				break;
		}
	}

	classification_ = numStable; // Assume it is a multiplicity of stables
	if ( numSolutions == ( numOther + numSingleZero ) )
		classification_ = 1; // Single stable
	if ( numSolutions == 1 && ( numSaddle == 1 || numOsc == 1 ) )
		classification_ = 0; // Oscillatory
	else if ( numSolutions >= 1 && numStable == 0 && 
			( numSolutions != (numSingleZero + numOther) ) )
		classification_ = 8; // Ill defined
	if ( numSolutions == 3 && numOsc == 1 && numStable == 1 )
		classification_ = 9; // Osc next to a stable state.
}

// True if it works
bool StateScanner::stateSettle( Eref me, Id& cj, Id& ss )
{
	static const Finfo* startFinfo = 
			Cinfo::find( "ClockJob" )->findFinfo( "start" );
	static const Finfo* reinitFinfo = 
			Cinfo::find( "ClockJob" )->findFinfo( "reinit" );
	static const Finfo* settleFinfo = 
			Cinfo::find( "SteadyState" )->findFinfo( "settle" );
	static const Finfo* stateTypeFinfo = 
			Cinfo::find( "SteadyState" )->findFinfo( "stateType" );
	static const Finfo* statusFinfo = 
			Cinfo::find( "SteadyState" )->findFinfo( "solutionStatus" );

	if ( useReinit_ && useBufferDose_ )
		set( cj(), reinitFinfo );
	set< double >( cj(), startFinfo, settleTime_ );
	if ( useSS_ ) {
		set( ss(), settleFinfo );
		unsigned int status;
		get< unsigned int >( ss(), statusFinfo, status );
		if ( status != 0 ) { // Need to decide if to do fallback.
			// try running to steady state.
			// cout << "status = 0, skipping\n";
			return 0;
		}
	}

	// Get the state type:
	unsigned int stateType = 0;
	get< unsigned int >( ss.eref(), stateTypeFinfo, stateType );
	x_ = stateType;

	// Update the state array.
	send1< double >( me, pushSlot, x_ );

	// Update the molecule conc arrays.
	ProcInfoBase p( 0, settleTime_ );
	send1< ProcInfo >( me, processSlot, &p );
	return 1;
}

void StateScanner::checkIfUniqueState( Eref me )
{
	static const Finfo* outputFinfo = 
			Cinfo::find( "Table" )->findFinfo( "output" );
	static const Finfo* popFinfo = 
			Cinfo::find( "Table" )->findFinfo( "pop" );

	vector< string > names;
	vector< vector< double > > solutions;
	unsigned int n = getTables( me, names, solutions );
	// First table in this entry is that of the solution types.
	if ( n <= 1 )
		return;

	// The last entry on the stack of solutions has the latest state.
	// Need to scan through to compare it with all others. If unique,
	// keep it, otherwise pop it by decrementing 'output' field.
	unsigned int numMonitored = solutions.size();
	if ( numMonitored <= 1 )  {
		cout << "Error: StateScanner::checkIfUniqueState: No molecules monitored\n";
		return;
	}
	

	unsigned int numStates = solutions[0].size();

	if ( numStates <= 1 ) // first state is always unique!
		return;

	Id stateId( "/kinetics/scan/state" );
	int xdivs;
	get< int >( stateId(), "xdivs", xdivs );
	// cout << "numStates = " << numStates << "; numSolutions = " << numSolutions << ", xdivs = " << xdivs << endl;
	for ( unsigned int i = 0; i < numStates - 1; ++i ) {
		double sumsq = 0.0;
		// Note first entry is states vector.
		for ( unsigned int j = 1; j < numMonitored; ++j ) {
			double dx = solutions[j][i] - solutions[j][ numStates - 1];
			sumsq += dx * dx;
		}
		if ( sumsq < solutionSeparation_ ) {
			// Similar solution to an existing one. Discard.
			Conn* c = me->targets( "processSrc", me.i );
			vector< vector< double > > solutions;
			// Get vectors for each child table
			while ( c->good() ) {
				set< double >( c->target(), outputFinfo, double(numStates - 1));
				set( c->target(), popFinfo );
				c->increment();
			}
			send0( me, popSlot );
			delete c;
			return;
		}
	}
	// If it gets here, then the solution was novel and is retained.
}
