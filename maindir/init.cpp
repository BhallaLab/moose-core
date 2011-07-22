/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**		   Copyright (C) 2003-2007 NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/
/*******************************************************************
 * File:			init.cpp
 * Description:	  
 * Author:		  Subhasis Ray
 * E-mail:		  ray.subhasis@gmail.com
 * Created:		 2007-09-25 15:38:08
 ********************************************************************/

#include <iostream>
#include "../basecode/header.h"
#include "../basecode/moose.h"
#include "../element/Neutral.h"
#include "../shell/Shell.h"
#include "init.h"

#ifdef WIN32	      // True for Visual C++ compilers
#include <Windows.h>  // for Win32 Sleep function
#include <process.h>  // for getpid
#else                 // Else assume POSIX
#include <ctime>      // for the POSIX nanosleep function.
                      // (could not find nanosleep implementation for Windows)
#include <unistd.h>   // for gethostname, getpid.
#endif // _MSC_VER

#ifdef USE_MPI
#include <mpi.h>
#include "MuMPI.h"	// Provides MUSIC-compatible MPI calls
#endif // USE_MPI
extern const Cinfo ** initCinfos();
extern bool setupProxyMsg(
	unsigned int srcNode, Id proxy, unsigned int srcFuncId,
	unsigned int proxySize,
	Id dest, int destMsg );
#ifdef PYMOOSE
extern void initPyMoose();
#endif
static Element* pj = 0;
static const Finfo* stepFinfo;
static const Finfo* reinitClockFinfo;
unsigned int init( int& argc, char**& argv )
{
    static bool inited = false;
    if (inited){
#ifndef NDEBUG
        cout << "Already initialized." << endl;
        return 0;
#endif
    }
	initMPI( argc, argv );
	initMoose( argc, argv );
	initParCommunication();
	initSched();
	initParSched();
	initGlobals();
// #ifdef PYMOOSE
//         initPyMoose();
// #endif
	doneInit();
        inited = true;
	return 0;
}

/**
 * Initializes MPI.
 * 
 * MPI::Init() takes references to 'argc' and 'argv' and consumes the arguments
 * meant for the mpi launcher. Moose will thus receive only those arguments meant
 * for it. Hence this call should be made before any argument processing.
 * 
 * If the last argument to moose is "-m" then each process will print its
 * process id and process 0 will wait for keyboard input. Useful for attaching a
 * debugger to the processes when running in parallel.
 * 
 * If the last argument is "-mi" then an infinite loop is used to pause, instead
 * of waiting for keyboard input. Sometimes the program refuses to wait for
 * keyboard input (happens when running moose using MUSIC--not sure why). The
 * infinite loop comes handy then.
 */
void initMPI( int& argc, char**& argv )
{
#ifdef USE_MPI
	MuMPI::Init( argc, argv );
	
	unsigned int numNodes = MuMPI::INTRA_COMM().Get_size();
	unsigned int myNode = MuMPI::INTRA_COMM().Get_rank();
	
	// If not done here, the Shell uses defaults suitable for one node.
	Shell::setNodes( myNode, numNodes );
	
	bool pause = strncmp( argv[ argc - 1 ], "-m", 2 ) == 0;
	bool infinite = strncmp( argv[ argc - 1 ], "-mi", 3 ) == 0;
	
	if ( pause ) {
		char hostname[ 256 ];
		// this loop will ensure that all nodes print the following message
		// in order of their rank
		for ( unsigned int i = 0; i < numNodes; i++ ) {
			if ( i == myNode ) {
				gethostname( hostname, sizeof( hostname ) );
				cout << "Rank " << myNode << " : PID " << getpid()
					 << " on " << hostname << " ready for attach" << endl;
			}
			MuMPI::INTRA_COMM().Barrier();
		}
		
		if ( myNode == 0 ) {
			if ( ! infinite ) {
				cout << "Paused, hit return to continue.\n" << flush;
				getchar();
			} else {
				cout <<
					"Paused. To resume, attach to process #0, and break out of "
					"the infinite loop. In GDB, use the command 'j +1'.\n" << flush;
				while( 1 );
			}
		}
		MuMPI::INTRA_COMM().Barrier();
	}
#endif // USE_MPI
}

/**
 * initMoose:
 *  - parses program arguments
 *  - sets up function ids through sortFuncVec
 *  - initializes root and shell
 */
void initMoose( int argc, char** argv )
{
	ArgParser::parseArguments(argc, argv);

	Property::initialize(ArgParser::getConfigFile(),Property::PROP_FORMAT);
        Property::addSimPath(ArgParser::getSimPath());

	cout << "SIMPATH = " << Property::getProperty(Property::SIMPATH) << endl;
	initCinfos();
	/**
	 * This function puts the FuncVecs in order and must be called
	 * after static initialization but before any messaging
	 */
	FuncVec::sortFuncVec();

	// This first call also initializes it. Not essential to do explicitly.
	Element::root();
	Id().setGlobal();

	Element* shell = Neutral::create( "Shell", "shell", Id(), Id::shellId() );
	Id::shellId().setNode( Shell::myNode() );
	// Id::shellId().setGlobal();

	assert( shell != 0 );
	assert( shell->id() == Id::shellId() );

	Id shellTest( "/shell" );
	assert( shellTest.good() );
	assert( shellTest == shell->id() );
}

/**
 * Sets up inter-node communication:
 *  - Creates postmasters
 *  - Connects shells across nodes (via the postmasters)
 */
void initParCommunication()
{
#ifdef USE_MPI
	unsigned int numNodes = Shell::numNodes();
	unsigned int myNode = Shell::myNode();
	Element* postmasters =
			Neutral::createArray( "PostMaster", "post", 
			Id::shellId(), Id::postId( 0 ), numNodes );
	
	assert( postmasters->numEntries() == numNodes );
	
	for ( unsigned int i = 0; i < numNodes; i++ ) {
		Eref pe = Eref( postmasters, i );
		set< unsigned int >( pe, "remoteNode", i );
	}

	// Here we set up the local connections between shell and postmasters.
	// These represent connections between the shell on this node and those on
	// other nodes.
	Eref shellE = Id::shellId().eref();
	Element* shell = shellE.e;
	const Finfo* parallelFinfo = shell->findFinfo( "parallel" );
	assert( parallelFinfo != 0 );

	for ( unsigned int i = 0; i < numNodes; i++ ) {
		if ( i != myNode) {
			bool ret = setupProxyMsg( i, 
				Id::shellId(), parallelFinfo->asyncFuncId(), 1,
				Id::shellId(), parallelFinfo->msg()
			);
			assert( ret != 0 );
			assert( Id::lastId().isProxy() );
			assert( Id::lastId().eref().data() == 
				Id::postId( i ).eref().data() ) ;
		}
	}
#endif
}

void initSched()
{
	/**
	 * Here we set up a bunch of predefined objects for scheduling, that
	 * exist simultaneously on each node.
	 */
	Element* sched =
		Neutral::create( "Neutral", "sched", Id(), Id::initId() );
	assert( sched != 0 );
	sched->id().setGlobal();

	Element* cj =
		Neutral::create( "ClockJob", "cj", sched->id(), Id::initId() );
	assert( cj != 0 );
        if (Property::getProperty(Property::AUTOSCHEDULE) == "false"){
            set<int>(cj, "autoschedule", 0);
        } else {
            set<int>(cj, "autoschedule", 1);
        }
	cj->id().setGlobal();
        
	// Not really honouring AUTOSCHEDULE setting -
	// May need only t0 for AUTOSCHEDULE=false
	// But creating a few extra clock ticks does not hurt as much as
	// not allowing user to change the clock settings
#ifdef USE_MPI
   	Element* t0 =
		Neutral::create( "ParTick", "t0", cj->id(), Id::initId() );
	assert( t0 != 0 );
	t0->id().setGlobal();

	Element* t1 =
		Neutral::create( "ParTick", "t1", cj->id(), Id::initId() );
	assert( t1 != 0 );
	t1->id().setGlobal();

	// pj declared at global scope in this file
	pj =
		Neutral::create( "ClockJob", "pj", sched->id(), Id::initId() );
	assert( pj != 0 );
	pj->id().setGlobal();

	cout << "pjid = " << pj->id() << endl;

	Element* pt0 =
			Neutral::create( "ParTick", "t0", pj->id(), Id::initId() );
	assert( pt0 != 0 );
	pt0->id().setGlobal();
#else
   	Neutral::create( "Tick", "t0", cj->id(), Id::newId() );
   	Neutral::create( "Tick", "t1", cj->id(), Id::newId() );
#endif // USE_MPI
}

/**
 * Initializes parallel scheduling.
 */
void initParSched()
{
#ifdef USE_MPI
	bool ret = 0;

	Id sched = Id::localId( "/sched" );
	assert( sched.good() );
	Id cj = Id::localId( "/sched/cj" );
	assert( cj.good() );
	Id t0id = Id::localId( "/sched/cj/t0" );
	assert( t0id.good() );
	Id t1id = Id::localId( "/sched/cj/t1" );
	assert( t1id.good() );
	Id pt0id = Id::localId( "/sched/pj/t0" );
	assert( pt0id.good() );
	Element* t0 = t0id.eref().e;
	Element* t1 = t1id.eref().e;
	Element* pt0 = pt0id.eref().e;

	ret = set< bool >( t0, "doSync", 1 ); 
	assert( ret );
	ret = set< bool >( t1, "doSync", 1 ); 
	assert( ret );
	ret = set< int >( t1, "stage", 1 ); 
	assert( ret );

	// set< int >( t0, "barrier", 1 ); // when running, ensure sync after t0
	// ensure sync for runtime clock ticks.
	// set< int >( pt0, "barrier", 1 ); // when running, ensure sync after t0

	///////////////////////////////////////////////////////////////////
	//	Here we connect up the postmasters to the ParTick.
	///////////////////////////////////////////////////////////////////
	Eref shellE = Id::shellId().eref();
	Element* shell = shellE.e;
	const Finfo* pollFinfo = shell->findFinfo( "pollSrc" );
	assert( pollFinfo != 0 );
	const Finfo* tickFinfo = t0->findFinfo( "parTick" );
	assert( tickFinfo != 0 );

	stepFinfo = pj->findFinfo( "step" );
	reinitClockFinfo = pj->findFinfo( "reinitClock" );
	assert( stepFinfo != 0 );

	ret = shellE.add( "pollSrc", pj, "step" );
	assert( ret );

	vector< Element* >::iterator j;
	Eref pe = Id::postId( Id::AnyIndex ).eref();
	ret = Eref( t0 ).add( "parTick", pe , "parTick",
		ConnTainer::One2All );
	assert( ret );
	ret = Eref( t1 ).add( "parTick", pe , "parTick",
		ConnTainer::One2All );
	assert( ret );
	ret = Eref( pt0 ).add( "parTick", pe, "parTick",
		ConnTainer::One2All );
	assert( ret );

	// cout << "On " << myNode << ", shell: " << shell->name() << endl;
	// shell->dumpMsgInfo();
	set( cj.eref(), "resched" );
	set( pj, "resched" );
	set( cj.eref(), "reinit" );
	set( pj, "reinit" );
#endif
}

void initGlobals()
{
	Element* lib = 
		Neutral::create( "Neutral", "library", Id(), Id::initId() );
	assert( lib != 0 );
	lib->id().setGlobal();

	Element* proto = 
		Neutral::create( "Neutral", "proto", Id(), Id::initId() );
	assert( proto != 0 );
	proto->id().setGlobal();

#ifdef USE_MUSIC
	Element* music =
		Neutral::create( "Music", "music", Id(), Id::initId() );
	assert( music != 0 );
	music->id().setGlobal();
#endif // USE_MUSIC
}

void doneInit()
{
#ifdef USE_MPI
	MuMPI::INTRA_COMM().Barrier();
	if ( Shell::myNode() == 0 )
		cout << "\nInitialized " << Shell::numNodes() << " nodes" << endl;

	for ( unsigned int i = 0; i < Shell::numNodes(); i++ ) {
		if ( i == Shell::myNode() ) {
			cout << "Node " << i << ":\n";
			Id::dumpState( cout );
		}
		MuMPI::INTRA_COMM().Barrier();
	}
#endif // USE_MPI
}

void setupDefaultSchedule(
	Element* t0, Element* t1, Element* cj)
{
	set< double >( t0, "dt", 1e-2 );
	set< double >( t1, "dt", 1e-2 );
	set< int >( t1, "stage", 1 );
	set( cj, "resched" );
	set( cj, "reinit" );
}

void terminateMPI( unsigned int myNode )
{
#ifdef USE_MPI
	/*
	Eref shell = Id::shellId().eref();
	if ( myNode != 0 ) {
		bool ret = set( shell, "poll" );
		assert( ret );
	}
	*/
	cout << myNode << "." << flush;
	MuMPI::INTRA_COMM().Barrier();
	MuMPI::Finalize();
#endif // USE_MPI
}

/// Portable sleep: Uses nanosleep for POSIX systems, and the Win32 Sleep function for MSVC++ compilers
void psleep( unsigned int nanoseconds )
{
#ifdef WIN32 // If this is an MS VC++ compiler..
	unsigned int milliseconds = nanoseconds / 1000000;
	Sleep( milliseconds );
#else           // else assume POSIX compliant..
	struct timespec ts = { 0, nanoseconds };
	nanosleep( &ts, 0 );
#endif // _MSC_VER
}

void pollPostmaster()
{
	unsigned int duration = 10000000; // 10000000 nsec, 10 msec.
	
	if ( pj != 0 ) {
		/*
		if ( Shell::numNodes() > 1 )
			cout << "Polling postmaster on node " << Shell::myNode() << endl;
		*/
		bool ret;
		
		// Reinit clockjob and ticks to their initial state
		ret = set( pj, reinitClockFinfo );
		assert( ret );
		
		// Take one step
		ret = set< int >( pj, stepFinfo, 1 );
		assert( ret );
		
		psleep( duration );
	}
}

#ifndef USE_MPI
void testParMsgOnSingleNode()
{
	// Dummy. The actual routine is in parallel/PostMaster.cpp.
}
void testPostMaster()
{
	// Dummy. The actual routine is in parallel/PostMaster.cpp.
}
#endif
