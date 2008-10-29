/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2008 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include <utility/utility.h>
#include <time.h> // for nanosleep. This is POSIX, so should be available
              // even from Windows.
int nanosleep(const struct timespec *rqtp, struct timespec *rmtp);

#ifdef USE_MPI
#include <mpi.h>
#include "MuMPI.h"	// Provides MUSIC-compatible MPI calls
#endif // USE_MPI

#include "../basecode/moose.h"
#include "../element/Neutral.h"
#include "../shell/Shell.h"

extern bool setupProxyMsg(
	unsigned int srcNode, Id proxy, unsigned int srcFuncId,
	unsigned int proxySize,
	Id dest, int destMsg );


using namespace std;

static Element* pj = 0;
static const Finfo* stepFinfo;

/**
 * Initializes MPI as well as scheduling and cross-node shell messaging.
 * Returns node number.
 */
unsigned int initMPI( int& argc, char**& argv )
{
#ifdef USE_MPI
	MuMPI::Init( argc, argv );
	
	unsigned int totalNodes = MuMPI::INTRA_COMM().Get_size();
	unsigned int myNode = MuMPI::INTRA_COMM().Get_rank();

	// If not done here, the Shell uses defaults suitable for one node.
	Shell::setNodes( myNode, totalNodes );

#ifdef USE_MUSIC
	if ( strncmp( argv[ argc - 1 ], "-m", 2 ) == 0 ) {
		char hostname[ 256 ];
		gethostname( hostname, sizeof( hostname ) );
		for ( unsigned int i = 0; i < totalNodes; i++ ) {
			if ( i == myNode ) {
				if ( argc >= 2 )
					cout << "Script argument: " << argv[ 1 ] << "\t: ";

				cout << "Rank " << myNode << " : PID " << getpid()
					 << " on " << hostname << " ready for attach" << endl;
			}
			MuMPI::INTRA_COMM().Barrier();
		}
		
		if ( myNode == 0 ) {
			int i = 0;
			while ( i == 0 );
				sleep( 5 );
		}
		MuMPI::INTRA_COMM().Barrier();
	}
#else // USE_MUSIC
	if ( argc >= 2 && strncmp( argv[1], "-m", 2 ) == 0 ) {

		char hostname[ 256 ];
		gethostname( hostname, sizeof( hostname ) );
		for ( unsigned int i = 0; i < totalNodes; i++ ) {
			if ( i == myNode ) {
				cout << "Rank " << myNode << " : PID " << getpid()
				     << " on " << hostname << " ready for attach" << endl;
			}
			MuMPI::INTRA_COMM().Barrier();
		}

		if ( myNode == 0 ) {
			cout << "Paused, hit return to continue" << flush;
			getchar();
		}
		MuMPI::INTRA_COMM().Barrier();
	}
#endif // USE_MUSIC

	return myNode;
#else // USE_MPI
	return 0;
#endif // USE_MPI
}


/**
 * Initializes parallel scheduling.
 */
void initParSched()
{
#ifdef USE_MPI
	unsigned int totalnodes = Shell::numNodes();
	unsigned int myNode = Shell::myNode();
	bool ret = 0;

	// Breakpoint for parallel debugging

	Element* postmasters =
			Neutral::createArray( "PostMaster", "post", 
			Id::shellId(), Id::postId( 0 ), totalnodes );
	assert( postmasters->numEntries() == totalnodes );
	// cerr << myNode << ".2b\n";
	for ( unsigned int i = 0; i < totalnodes; i++ ) {
		Eref pe = Eref( postmasters, i );
		set< unsigned int >( pe, "remoteNode", i );
	}
	Id sched = Id::localId( "/sched" );
	assert( sched.good() );
	Id cj = Id::localId( "/sched/cj" );
	assert( cj.good() );
	Id t0id = Id::localId( "/sched/cj/t0" );
	assert( t0id.good() );
	Id t1id = Id::localId( "/sched/cj/t1" );
	assert( t1id.good() );
	Element* t0 = t0id.eref().e;
	Element* t1 = t1id.eref().e;

	ret = set< bool >( t0, "doSync", 1 ); 
	assert( ret );
	ret = set< bool >( t1, "doSync", 1 ); 
	assert( ret );
	ret = set< int >( t1, "stage", 1 ); 
	assert( ret );
	// cerr << myNode << ".2c\n";
	// This one handles parser and postmaster scheduling.
	/*
	Id t0id = Id::localId( "/sched/cj/t0" );
	if ( t0id.good() ) {
		set( t0id(), "destroy" );
	}
	Element* t0 =
			Neutral::create( "ParTick", "t0", cj, Id::scratchId() );
	Element* t1 =
			Neutral::create( "ParTick", "t1", cj, Id::scratchId() );
	*/
	pj = Neutral::create( "ClockJob", "pj", sched, Id::scratchId() );
	assert( pj != 0 );
	cout << "pjid = " << pj->id() << endl;
	// set< int >( t0, "barrier", 1 ); // when running, ensure sync after t0
	
	// ensure sync for runtime clock ticks.

	Element* pt0 =
			Neutral::create( "ParTick", "t0", pj->id(), Id::scratchId() );
	assert( pt0 != 0 );
	// set< int >( pt0, "barrier", 1 ); // when running, ensure sync after t0

	///////////////////////////////////////////////////////////////////
	//	Here we connect up the postmasters to the shell and the ParTick.
	///////////////////////////////////////////////////////////////////
	Eref shellE = Id::shellId().eref();
	Element* shell = shellE.e;
	const Finfo* parallelFinfo = shell->findFinfo( "parallel" );
	assert( parallelFinfo != 0 );
	const Finfo* pollFinfo = shell->findFinfo( "pollSrc" );
	assert( pollFinfo != 0 );
	const Finfo* tickFinfo = t0->findFinfo( "parTick" );
	assert( tickFinfo != 0 );

	stepFinfo = pj->findFinfo( "step" );
	assert( stepFinfo != 0 );

	SetConn c( shellE );
	// Here we need to set up the local connections that will become
	// connections between shell on this node to all other nodes

	for ( unsigned int i = 0; i < totalnodes; i++ ) {
		if ( i != myNode) {
			ret = setupProxyMsg( i, 
				Id::shellId(), parallelFinfo->asyncFuncId(), 1,
				Id::shellId(), parallelFinfo->msg()
			);
			assert( ret != 0 );
			assert( Id::lastId().isProxy() );
			assert( Id::lastId().eref().data() == 
				Id::postId( i ).eref().data() ) ;
		}
	}

	ret = shellE.add( "pollSrc", pj, "step" );
	assert( ret );

	vector< Element* >::iterator j;
	ret = shellE.add( "pollSrc", pj, "step" );
	// ret = pollFinfo->add( shell, pj, pj->findFinfo( "step" ) );
	assert( ret );
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

	// cerr << myNode << ".2d\n";

	MuMPI::INTRA_COMM().Barrier();
	if ( myNode == 0 )
		cout << "\nInitialized " << totalnodes << " nodes\n";
	MuMPI::INTRA_COMM().Barrier();
#endif // USE_MPI
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

void pollPostmaster()
{
	
  
	//static struct timespec ts( 0, 10000000L );
    //static struct timespec ts=static struct timespec( 0, 10000000L );
//timespec ts1= timespec( 0, 10000000L );
	static struct timespec ts = { 0, 10000000L }; // 10000000 nsec, 10 msec.
	if ( pj != 0 ) {
		/*
		if ( Shell::numNodes() > 1 )
			cout << "Polling postmaster on node " << Shell::myNode() << endl;
			*/
		bool ret = set< int >( pj, stepFinfo, 1 );
		assert( ret );
		nanosleep( &ts, 0 );
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
