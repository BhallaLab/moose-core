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
#include "../element/Neutral.h"
/*
#include "RateTerm.h"
#include "KinSparseMatrix.h"
#include "InterSolverFlux.h"
#include "Stoich.h"
*/
#include "StateScanner.h"

const Cinfo* initStateScannerCinfo()
{
	/**
	 * This picks up the entire Stoich data structure
	static Finfo* gslShared[] =
	{
		new SrcFinfo( "reinitSrc", Ftype0::global() ),
		new DestFinfo( "assignStoich",
			Ftype1< void* >::global(),
			RFCAST( &StateScanner::assignStoichFunc )
			),
		new DestFinfo( "setMolN",
			Ftype2< double, unsigned int >::global(),
			RFCAST( &StateScanner::setMolN )
			),
	};
	 */

	/**
	 * This controls the stack operations of the child Table objects
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
		/*
		new ValueFinfo( "numTrackedMolecules", 
			ValueFtype1< int >::global(),
			GFCAST( &StateScanner::setNumTrackedMolecules ), 
			RFCAST( &StateScanner::getNumTrackedMolecules ),
			"This is the size of the trackedMolecules array."
		),
		new LookupFinfo( "trackedMolecules",
			LookupFtype< Id, unsigned int >::global(),
				GFCAST( &StateScanner::getTrackedMolecule ),
				RFCAST( &StateScanner::setTrackedMolecule ), // dummy
				"Molecules to track during dose-response or state classification"
		),
		*/
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

static const Slot procSlot =
	initStateScannerCinfo()->getSlot( "processSrc.pushSrc" );

static const Slot reinitSlot =
	initStateScannerCinfo()->getSlot( "processSrc.reinitSrc" );

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
		numOther_( 0 )
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

const string& uniqueName( Id elm )
{
	return elm()->name();
}

/** 
 * Creates a table and connects to it by a process message, and 
 * connects the table itself to the tracked molecule
 */
void StateScanner::addTrackedMolecule( const Conn* c, Id val )
{
	Eref scanner = c->target();
	string name = uniqueName( val );
	bool ret;
	Element* tab = Neutral::create( "Table", name, scanner.id(),
		Id::scratchId() );
	Eref( tab ).dropAll( "process" ); // Eliminate the default msg.
	assert( tab->id().good() );

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
		variableMol, start, end, numSteps, 0 );
}

void StateScanner::logDoseResponse( const Conn* c,
	Id variableMol,
	double start, double end,
	unsigned int numSteps )
{
	static_cast< StateScanner* >( c->data() )->innerDoseResponse(
		variableMol, start, end, numSteps, 1 );
}

void StateScanner::innerDoseResponse( Id variableMol,
	double start, double end,
	unsigned int numSteps,
	bool useLog)
{
	;
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

