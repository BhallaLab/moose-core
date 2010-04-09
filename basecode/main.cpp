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
#ifdef USE_MPI
#include <mpi.h>
#endif

extern void testSync();
extern void testAsync();
extern void testSyncArray( unsigned int size, unsigned int numThreads,
	unsigned int method );
extern void testShell();
extern void testScheduling();

Id init( int argc, char** argv )
{
	int numCores = 1;
	int numNodes = 1;
	int myNode = 0;
	bool isSingleThreaded = 0;
	int opt;
	while ( ( opt = getopt( argc, argv, "shn:c:" ) ) != -1 ) {
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
			case 'h': // help
			default:
				cout << "Usage: moose -singleThreaded -help -c numCores -n numNodes\n";
				exit( 1 );
		}
	}
#ifdef USE_MPI
	int provided;
	MPI_Init_thread( &argc, &argv, MPI_THREAD_SERIALIZED, &provided );
	if ( provided < MPI_THREAD_SERIALIZED ) {
		cout << "Warning: This MPI implementation does not like multithreading: " << provided << "\n";
	}

	MPI_Comm_size( MPI_COMM_WORLD, &numNodes );
	MPI_Comm_rank( MPI_COMM_WORLD, &myNode );
	// myNode = MPI::COMM_WORLD.Get_rank();
	cout << "on node " << myNode << ", numNodes = " << numNodes << endl;
#endif

	Msg::initNull();
	Id shellId = Id::nextId();
	Element* shelle = new Element( shellId, Shell::initCinfo(), "root", 1 );
	// Shell::initCinfo()->create( shellId, "root", 1 );

	Id clockId = Id::nextId();
	// Clock::initCinfo()->create( clockId, "clock", 1 );
	Element* clocke = new Element( clockId, Clock::initCinfo(), "clock", 1 );
	// Should put this initialization stuff within the Clock creation
	// step. This means I need to add an optional init func into the Cinfo
	// constructor, or to add the init func as a virtual func in Data.
	FieldElement< Tick, Clock, &Clock::getTick >* ticke =
		new FieldElement< Tick, Clock, &Clock::getTick >
		( 
			Tick::initCinfo(), clocke,
			&Clock::getNumTicks, &Clock::setNumTicks 
		);
	Id tickId = Id::nextId();
	tickId.bindIdToElement( ticke );

	assert ( shellId == Id() );
	assert( clockId == Id( 1 ) );
	assert( tickId == Id( 2 ) );
	SetGet::setShell();
	Shell* s = reinterpret_cast< Shell* >( shellId.eref().data() );
	s->setHardware( isSingleThreaded, numCores, numNodes, myNode );
	s->loadBalance();
	Shell::connectMasterMsg();
	Msg* m = new OneToOneMsg( shelle, shelle );
	assert ( m != 0 );

	return shellId;
}

int main( int argc, char** argv )
{
	Id shellId = init( argc, argv );
#ifdef DO_UNIT_TESTS
#endif
	cout << "testing: ";
	testAsync();
//	testScheduling();
	testShell();
	cout << endl;

	// Note that the main loop remains the parser loop, though it may
	// spawn a lot of other stuff.
	Element* shelle = shellId();
	Shell* s = reinterpret_cast< Shell* >( shelle->data( 0 ) );
	ProcInfo p;
	// Actually here we should launch off the thread doing
	// Shell messaging/MPI, and yield control to the parser.
	if ( s->myNode() == 0 )
		s->launchParser();
	else
		s->launchMsgLoop( shelle );

	// cout << s->myNode() << ": Main: out of parser/MsgLoop\n";

	shellId.destroy();
	Id(1).destroy();
	Id(2).destroy();
#ifdef USE_MPI
	MPI_Finalize();
#endif
	return 0;
}

