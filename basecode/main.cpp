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
#include "../scheduling/TickMgr.h"
#include "../scheduling/TickPtr.h"
#include "../scheduling/Clock.h"
#include "DiagonalMsg.h"
#include "SparseMatrix.h"
#include "SparseMsg.h"
#ifdef USE_MPI
#include <mpi.h>
#endif
#include "../shell/Shell.h"
#ifdef MACOSX
#include <sys/sysctl.h>
#endif // MACOSX

#ifdef DO_UNIT_TESTS
extern void testSync();
extern void testAsync();
extern void testSyncArray( unsigned int size, unsigned int numThreads,
	unsigned int method );
extern void testShell();
extern void testScheduling();
extern void testSchedulingProcess();
extern void testBuiltins();
extern void testBuiltinsProcess();

extern void testMpiScheduling();
extern void testMpiBuiltins();
extern void testMpiShell();
extern void testMsg();
extern void testMpiMsg();
extern void testKinetics();
extern void testBiophysics();
extern void testBiophysicsProcess();
extern void testKineticsProcess();
extern void testGeom();
extern void testMesh();

extern void initMsgManagers();
extern void destroyMsgManagers();
void regressionTests();
#endif
extern void speedTestMultiNodeIntFireNetwork( 
	unsigned int size, unsigned int runsteps );

#ifdef USE_SMOLDYN
	extern void testSmoldyn();
#endif
bool benchmarkTests( int argc, char** argv );

//////////////////////////////////////////////////////////////////
// System-dependent function here
//////////////////////////////////////////////////////////////////

unsigned int getNumCores()
{
	unsigned int numCPU = 0;
#ifdef WIN_32
	SYSTEM_INFO sysinfo;
	GetSystemInfo( &sysinfo );

	numCPU = sysinfo.dwNumberOfProcessors;
#endif

#ifdef LINUX
	numCPU = sysconf( _SC_NPROCESSORS_ONLN );
#endif

#ifdef MACOSX
	nt mib[4];
	size_t len; 

	/* set the mib for hw.ncpu */
	mib[0] = CTL_HW;
	mib[1] = HW_AVAILCPU;  // alternatively, try HW_NCPU;

	/* get the number of CPUs from the system */
	sysctl(mib, 2, &numCPU, &len, NULL, 0);

	if( numCPU < 1 ) 
	{
		mib[1] = HW_NCPU;
		sysctl( mib, 2, &numCPU, &len, NULL, 0 );
	}
#endif
	if ( numCPU < 1 )
	{
		cout << "No CPU information available. Assuming single core." << endl;
		numCPU = 1;
	}
	return numCPU;
}

bool quitFlag = 0;
//////////////////////////////////////////////////////////////////

void checkChildren( Id parent, const string& info )
{
	vector< Id > ret;
	Neutral::children( parent.eref(), ret );
	cout << info << " checkChildren of " << parent()->getName() << ": " <<
		ret.size() << " children\n";
	for ( vector< Id >::iterator i = ret.begin(); i != ret.end(); ++i )
	{
		cout << (*i)()->getName() << endl;
	}
}

Id init( int argc, char** argv )
{
	int numCores = getNumCores();
	int numThreads = numCores;
	int numNodes = 1;
	int myNode = 0;
	bool isSingleThreaded = 0;
	bool isInfinite = 0;
	int opt;
#ifdef USE_MPI
	int provided;
	// OpenMPI does not use argc or argv.
	// unsigned int temp_argc = 1;
	//MPI_Init_thread( &temp_argc, &argv, MPI_THREAD_SERIALIZED, &provided );
	MPI_Init_thread( &argc, &argv, MPI_THREAD_SERIALIZED, &provided );

	MPI_Comm_size( MPI_COMM_WORLD, &numNodes );
	MPI_Comm_rank( MPI_COMM_WORLD, &myNode );
	if ( provided < MPI_THREAD_SERIALIZED && myNode == 0 ) {
		cout << "Warning: This MPI implementation does not like multithreading: " << provided << "\n";
	}
	// myNode = MPI::COMM_WORLD.Get_rank();
#endif
	/**
	 * Here we allow the user to override the automatic identification
	 * of processor configuration
	 */
	while ( ( opt = getopt( argc, argv, "shiqn:t:b:B:" ) ) != -1 ) {
		switch ( opt ) {
			case 's': // Single threaded mode. Overrides numThreads.
				isSingleThreaded = 1;
				break;
			case 'i' : // infinite loop, used for multinode debugging, to give gdb something to attach to.
				isInfinite = 1;
				break;
			case 'n': // Multiple nodes
				numNodes = atoi( optarg );
				break;
			case 't': // Number of process threads to use. Default is
				// obtained from hardware NumCores.
				numThreads = atoi( optarg );
				break;
			case 'b': // Benchmark: handle later.
				break;
			case 'B': // Benchmark, dump data: handle later.
				break;
			case 'q': // quit immediately after completion.
				quitFlag = 1;
				break;
			case 'h': // help
			default:
				cout << "Usage: moose -singleThreaded -help -infiniteLoop -quit -t numComputeThreads -n numNodes -benchmark [ksolve intFire hhNet]\n";
				exit( 1 );
		}
	}
	if ( isSingleThreaded )
		numThreads = 0;
	if ( myNode == 0 ) 
		cout << "on node " << myNode << ", numNodes = " << numNodes << ", numCores = " << numCores << ", numComputeThreads = " << numThreads << endl;

	Msg::initNull();
	Id shellId;
	vector< DimInfo > dims;
	Element* shelle = 
		new Element( shellId, Shell::initCinfo(), "root", dims, 0, 1 );
	// Shell::initCinfo()->create( shellId, "root", 1 );

	Id clockId = Id::nextId();

	Shell* s = reinterpret_cast< Shell* >( shellId.eref().data() );
	s->setShellElement( shelle );
	s->setHardware( numThreads, numCores, numNodes, myNode );
	s->loadBalance();

	// Element* clocke = 
	new Element( clockId, Clock::initCinfo(), "clock", dims, 1, 1 );

	// Clock::initCinfo()->postCreationFunc( clockId, clocke );
	// Should put this initialization stuff within the Clock creation
	// step. This means I need to add an optional init func into the Cinfo
	// constructor, or to add the init func as a virtual func in Data.
	Id tickId( 2 );
	assert( tickId() != 0 );
	assert( tickId.value() == 2 );
	assert( tickId()->getName() == "tick" ) ;

	Id classMasterId( 3 );
	new Element( classMasterId, Neutral::initCinfo(), "classes", dims, 1, 1 );

	assert ( shellId == Id() );
	assert( clockId == Id( 1 ) );
	assert( tickId == Id( 2 ) );
	assert( classMasterId == Id( 3 ) );


	/// Sets up the Elements that represent each class of Msg.
	Msg::initMsgManagers();

	s->connectMasterMsg();

	Shell::adopt( shellId, clockId );
	Shell::adopt( shellId, classMasterId );

	Cinfo::makeCinfoElements( classMasterId );


	// This will be initialized within the Process loop, and better there
	// as it flags attempts to call the Reduce operations before ProcessLoop
	// Qinfo::clearReduceQ( numCores ); // Initialize the ReduceQ entry.


	// SetGet::setShell();
	// Msg* m = new OneToOneMsg( shelle, shelle );
	// assert ( m != 0 );
	
	while ( isInfinite ) // busy loop for debugging under gdb and MPI.
		;

	return shellId;
}

/**
 * These tests are meant to run on individual nodes, and should
 * not invoke MPI calls. They should not be run when MPI is running.
 * These tests do not use the threaded/MPI event loop and are the most
 * basic of the set.
 */
void nonMpiTests( Shell* s )
{
#ifdef DO_UNIT_TESTS
	if ( Shell::myNode() == 0 ) {
		unsigned int numNodes = s->numNodes();
		unsigned int numCores = s->numCores();
		unsigned int numThreads = s->numProcessThreads();
		if ( numCores > 0 )
		// s->setHardware( isSingleThreaded, numCores, numNodes, myNode );
		s->setHardware( 0, 1, 1, 0 );
		testAsync();
		testMsg();
		testShell();
		testScheduling();
		testBuiltins();
		testKinetics();
		testBiophysics();
		testGeom();
		testMesh();
#ifdef USE_SMOLDYN
		testSmoldyn();
#endif
		s->setHardware( numThreads, numCores, numNodes, 0 );
	}
#endif
}

/**
 * These tests involve the threaded/MPI process loop and are the next
 * level of tests.
 */
void processTests( Shell* s )
{
#ifdef DO_UNIT_TESTS
	testSchedulingProcess();
	testBuiltinsProcess();
	testKineticsProcess();
	testBiophysicsProcess();
#endif
}

/**
 * These are tests that are MPI safe. They should also run
 * properly on single nodes.
 */
void mpiTests()
{
#ifdef DO_UNIT_TESTS
		testMpiMsg();
		cout << "." << flush;
		testMpiShell();
		cout << "." << flush;
		testMpiBuiltins();
		cout << "." << flush;
		testMpiScheduling();
		cout << "." << flush;
#endif
}
#ifndef PYMOOSE
int main( int argc, char** argv )
{
	Id shellId = init( argc, argv );
	// Note that the main loop remains the parser loop, though it may
	// spawn a lot of other stuff.
	Element* shelle = shellId();
	Shell* s = reinterpret_cast< Shell* >( shelle->dataHandler()->data( 0 ) );
	nonMpiTests( s ); // These tests do not need the process loop.

	if ( !s->isSingleThreaded() ) {
		Qinfo::initMutex(); // Mutex used to align Parser and MOOSE threads.
		Shell::launchThreads(); // Here we set off the thread/MPI process loop.
	}
	if ( Shell::myNode() == 0 ) {
#ifdef DO_UNIT_TESTS
		mpiTests();
		processTests( s );
		regressionTests();
#endif
		// These are outside unit tests because they happen in optimized
		// mode, using a command-line argument. As soon as they are done
		// the system quits, in order to estimate timing.
		if ( benchmarkTests( argc, argv ) || quitFlag )
			s->doQuit();
		else 
			Shell::launchParser(); // Here we set off a little event loop to poll user input. It deals with the doQuit call too.
	}
	
	// Somehow we need to return control to our parser. Then we clean up
	if ( !Shell::isSingleThreaded() ) {
		Shell::joinThreads();
		Qinfo::freeMutex();
	}
	Neutral* ns = reinterpret_cast< Neutral* >( shelle->dataHandler()->data( 0 ) );
	ns->destroy( shellId.eref(), 0, 0 );
#ifdef USE_MPI
	MPI_Finalize();
#endif
	return 0;
}
#endif

