/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2009 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
#include <sys/time.h>
#include <math.h>
#include <queue>
#include <unistd.h> // for getopt
#include "../scheduling/Tick.h"
#include "../scheduling/TickPtr.h"
#include "../scheduling/Clock.h"
#include "Neutral.h"
#include "DiagonalMsg.h"
#include "SparseMatrix.h"
#include "SparseMsg.h"
#include "PsparseMsg.h"
#include "AssignmentMsg.h"
#include "AssignVecMsg.h"
#ifdef USE_MPI
#include <mpi.h>
#endif

extern void testSync();
extern void testAsync();
extern void testSyncArray( unsigned int size, unsigned int numThreads,
	unsigned int method );
extern void testShell();
extern void testScheduling();
extern void testBuiltins();

extern void testMpiScheduling();
extern void testMpiBuiltins();
extern void testMpiShell();

Id SingleMsg::id_;
Id OneToOneMsg::id_;
Id OneToAllMsg::id_;
Id DiagonalMsg::id_;
Id SparseMsg::id_;
Id PsparseMsg::id_;
Id AssignmentMsg::id_;
Id AssignVecMsg::id_;

void initMsgManagers()
{
	vector< unsigned int > dims( 1, 2 );

	// This is to be the parent of al the msg managers.
	Id msgManagerId = Id::nextId();
	new Element( msgManagerId, Neutral::initCinfo(), "Msgs", dims, 1 );

	SingleMsg::id_ = Id::nextId();
	new Element( SingleMsg::id_, SingleMsgWrapper::initCinfo(), "singleMsg", dims, 1 );

	OneToOneMsg::id_ = Id::nextId();
	new Element( OneToOneMsg::id_, OneToOneMsgWrapper::initCinfo(), "oneToOneMsg", dims, 1 );

	OneToAllMsg::id_ = Id::nextId();
	new Element( OneToAllMsg::id_, SingleMsgWrapper::initCinfo(), "oneToAllMsg", dims, 1 );
	DiagonalMsg::id_ = Id::nextId();
	new Element( DiagonalMsg::id_, SingleMsgWrapper::initCinfo(), "diagonalMsg", dims, 1 );
	SparseMsg::id_ = Id::nextId();
	new Element( SparseMsg::id_, SingleMsgWrapper::initCinfo(), "sparseMsg", dims, 1 );
	PsparseMsg::id_ = Id::nextId();
	new Element( PsparseMsg::id_, SingleMsgWrapper::initCinfo(), "pSparseMsg", dims, 1 );
	AssignmentMsg::id_ = Id::nextId();
	new Element( AssignmentMsg::id_, SingleMsgWrapper::initCinfo(), "assignmentMsg", dims, 1 );
	AssignVecMsg::id_ = Id::nextId();
	new Element( AssignVecMsg::id_, SingleMsgWrapper::initCinfo(), "assignVecMsg", dims, 1 );
}

Id init( int argc, char** argv )
{
	int numCores = 1;
	int numNodes = 1;
	int myNode = 0;
	bool isSingleThreaded = 0;
	bool isInfinite = 0;
	int opt;
	while ( ( opt = getopt( argc, argv, "shin:c:" ) ) != -1 ) {
		switch ( opt ) {
			case 's': // Single threaded mode
				isSingleThreaded = 1;
				break;
			case 'c': // Multiple cores per node
				// Each node handles 
				numCores = atoi( optarg );
				break;
			case 'n': // Multiple nodes
				numNodes = atoi( optarg );
				break;
			case 'i' : // infinite loop, used for multinode debugging, to give gdb something to attach to.
				isInfinite = 1;
				break;
			case 'h': // help
			default:
				cout << "Usage: moose -singleThreaded -help -infiniteLoop -c numCores -n numNodes\n";
				exit( 1 );
		}
	}
#ifdef USE_MPI
	int provided;
	MPI_Init_thread( &argc, &argv, MPI_THREAD_SERIALIZED, &provided );

	MPI_Comm_size( MPI_COMM_WORLD, &numNodes );
	MPI_Comm_rank( MPI_COMM_WORLD, &myNode );
	if ( provided < MPI_THREAD_SERIALIZED && myNode == 0 ) {
		cout << "Warning: This MPI implementation does not like multithreading: " << provided << "\n";
	}
	// myNode = MPI::COMM_WORLD.Get_rank();
	// cout << "on node " << myNode << ", numNodes = " << numNodes << endl;
#endif

	Msg::initNull();
	Id shellId;
	vector< unsigned int > dims;
	dims.push_back( 1 );
	Element* shelle = 
		new Element( shellId, Shell::initCinfo(), "root", dims, 1 );
	// Shell::initCinfo()->create( shellId, "root", 1 );

	Id clockId = Id::nextId();
	// Clock::initCinfo()->create( clockId, "clock", 1 );
	// Element* clocke = 
		new Element( clockId, Clock::initCinfo(), "clock", dims, 1 );
	// Clock::initCinfo()->postCreationFunc( clockId, clocke );
	// Should put this initialization stuff within the Clock creation
	// step. This means I need to add an optional init func into the Cinfo
	// constructor, or to add the init func as a virtual func in Data.
	/*
	FieldElement< Tick, Clock, &Clock::getTick >* ticke =
		new FieldElement< Tick, Clock, &Clock::getTick >
		( 
			Tick::initCinfo(), clocke,
			&Clock::getNumTicks, &Clock::setNumTicks 
		);
	Id tickId = Id::nextId();
	tickId.bindIdToElement( ticke );
	*/
	Id tickId( 2 );
	assert( tickId() != 0 );
	assert( tickId()->name() == "tick" ) ;

	assert ( shellId == Id() );
	assert( clockId == Id( 1 ) );
	assert( tickId == Id( 2 ) );

	initMsgManagers();

	// SetGet::setShell();
	Shell* s = reinterpret_cast< Shell* >( shellId.eref().data() );
	s->setShellElement( shelle );
	s->setHardware( isSingleThreaded, numCores, numNodes, myNode );
	s->loadBalance();
	Shell::connectMasterMsg();
	// Msg* m = new OneToOneMsg( shelle, shelle );
	// assert ( m != 0 );
	
	while ( isInfinite ) // busy loop for debugging under gdb and MPI.
		;

	return shellId;
}

/**
 * These tests are meant to run on individual nodes, and should
 * not invoke MPI calls. They should not be run when MPI is running
 */
void nonMpiTests()
{
#ifdef DO_UNIT_TESTS
	if ( Shell::numNodes() == 1 ) {
		testAsync();
		testScheduling();
		testBuiltins();
		testShell();
	}
#endif
}

/**
 * These are tests that are MPI safe. They should also run
 * properly on single nodes.
 */
void mpiTests()
{
#ifdef DO_UNIT_TESTS
	// if ( Shell::numNodes() > 1 ) {
		testMpiShell();
		testMpiBuiltins();
		testMpiScheduling();
	// }
#endif
}

int main( int argc, char** argv )
{
	Id shellId = init( argc, argv );
	// Note that the main loop remains the parser loop, though it may
	// spawn a lot of other stuff.
	Element* shelle = shellId();
	Shell* s = reinterpret_cast< Shell* >( shelle->dataHandler()->data( 0 ) );
	nonMpiTests();
	// ProcInfo p;
	// Actually here we should launch off the thread doing
	// Shell messaging/MPI, and yield control to the parser.
	if ( s->myNode() == 0 ) {
		mpiTests();
		s->launchParser();
	} else {
		s->launchMsgLoop( shelle );
	}

	// cout << s->myNode() << ": Main: out of parser/MsgLoop\n";

	shellId.destroy();
	Id(1).destroy();
	Id(2).destroy();
#ifdef USE_MPI
	MPI_Finalize();
#endif
	return 0;
}

