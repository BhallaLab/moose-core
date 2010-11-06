/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include <unistd.h>
#include <mpi.h>
#include <stdlib.h>
#include "header.h"

using namespace std;
void runParserStuff( const ProcInfo* p );

void* reportGraphics( void* info )
{
	ProcInfo *p = reinterpret_cast< ProcInfo* >( info );
	cout << "reportGraphics on " << p->myNode << ":" << 
		p->threadIndexInGroup << endl;
	pthread_exit( NULL );
}

void launchThreads( int numNodes, int numCores, int myNode )
{
	pthread_attr_t attr;
	pthread_attr_init( &attr );
	pthread_attr_setdetachstate( &attr, PTHREAD_CREATE_JOINABLE );
	int numThreads = numCores + 1; // Add one for the MPI thread.

	// Extra thread on barrier 1 for parser control on node 0 
	// (the main thread here).
	int numBarrier1Threads = numThreads + ( myNode == 0 );

	FuncBarrier barrier1( numBarrier1Threads, &swapQ );
	FuncBarrier barrier2( numThreads, &swapMpiQ );
	pthread_barrier_t barrier3;
	int ret;
	pthread_t gThread;
	pthread_mutex_t shellSendMutex;
	pthread_cond_t parserBlockCond;

	ret = pthread_mutex_init( &shellSendMutex, NULL );
	assert( ret == 0 );

	ret = pthread_cond_init( &parserBlockCond, NULL );
	assert( ret == 0 );

	ret = pthread_barrier_init( &barrier3, NULL, numThreads );
	assert( ret == 0 );

	if ( myNode == 0 ) { // Launch graphics thread only on node 0.
		ProcInfo p;
		// pthread_barrier_t barrier1;
		int rc = pthread_create(&gThread, NULL, reportGraphics, 
			(void *)&p );
		if ( rc )
			cout << "Error: return code from pthread_create: " << rc << endl;
	}

	vector< ProcInfo > p( numBarrier1Threads );
	// An extra thread is used by MPI, and on node 0, yet another for Shell
	pthread_t* threads = new pthread_t[ numBarrier1Threads ];

	for ( int i = 0; i < numBarrier1Threads; ++i ) {
		// Note that here we put # of compute cores, not total threads.
		p[i].numThreadsInGroup = numCores; 

		p[i].threadIndexInGroup = i;
		p[i].myNode = myNode;
		p[i].numNodes = numNodes;
		p[i].barrier1 = &barrier1;
		p[i].barrier2 = &barrier2;
		p[i].barrier3 = &barrier3;
		p[i].shellSendMutex = &shellSendMutex;
		p[i].parserBlockCond = &parserBlockCond;

		if ( i < numCores ) { // These are the compute threads
			if ( myNode == 0 && i == 0 ) { // For now just set off rule 0
				Tracker t( numNodes, numCores, Rule( i % 4 ) );
				addToOutQ( &p[i], &t );
			}
			int rc = pthread_create( threads + i, NULL, eventLoop, 
				(void *)&p[i] );
			assert( rc == 0 );
		} else if ( i == numCores ) { // mpiThread stufff.
			int rc = pthread_create( 
				threads + i, NULL, mpiEventLoop, (void *)&p[i] );
			assert( rc == 0 );
		} else if ( i == numThreads ) { // shellThread stuff.
			int rc = pthread_create( 
				threads + i, NULL, shellEventLoop, (void *)&p[i] );
			assert( rc == 0 );
		}
	}

	if ( myNode == 0 )
			runParserStuff( &p[ numBarrier1Threads - 1 ] );

	// clean up. Add an extra time round loop for the MPI thread.
	for ( int i = 0; i < numBarrier1Threads; ++i ) {
		void* status;
		ret = pthread_join( threads[i], &status );
		if ( ret )
			cout << "Error: Unable to join threads\n";
	}

	if ( myNode == 0 ) { // clean up graphics thread only on node 0.
		void* status;
		ret = pthread_join( gThread, &status );
		if ( ret )
			cout << "Error: Unable to join threads\n";
	}

	delete[] threads;
	pthread_attr_destroy( &attr );
	ret = pthread_mutex_destroy( &shellSendMutex );
	ret = pthread_cond_destroy( &parserBlockCond );
	assert( ret == 0 );
}

void runParserStuff( const ProcInfo* p )
{
	assert( p->myNode == 0 );
	Tracker t( p->numNodes, p->numThreadsInGroup, raster90 );
	ProcInfo temp( *p );
	temp.threadIndexInGroup = 0;
	usleep( 500000 );
	pthread_mutex_lock( p->shellSendMutex );
		setBlockingParserCall( 1 );
		addToOutQ( &temp, &t ); // equivalent to 'send' call.
		while ( isAckPending() )
			pthread_cond_wait( p->parserBlockCond, p->shellSendMutex );
		setBlockingParserCall( 0 );
	pthread_mutex_unlock( p->shellSendMutex );
	usleep( 500000 );

	Tracker tend( p->numNodes, p->numThreadsInGroup, endit );
	pthread_mutex_lock( p->shellSendMutex );
		setBlockingParserCall( 1 );
		addToOutQ( &temp, &tend ); // equivalent to 'send' call.
		while ( isAckPending() )
			pthread_cond_wait( p->parserBlockCond, p->shellSendMutex );
		setBlockingParserCall( 0 );
	pthread_mutex_unlock( p->shellSendMutex );
	usleep( 500000 );
}

int main( int argc, char** argv )
{
	int numNodes = 1;
	int numCores = 1;
	int myNode = 0;
	int opt;

	int provided;
	MPI_Init_thread( &argc, &argv, MPI_THREAD_SERIALIZED, &provided );
	MPI_Comm_size( MPI_COMM_WORLD, &numNodes );
	MPI_Comm_rank( MPI_COMM_WORLD, &myNode );

	while ( ( opt = getopt( argc, argv, "n:c:" ) ) != -1 ) {
		switch ( opt ) {
			case 'n': // Num nodes
				numNodes = atoi( optarg );
				break;
			case 'c': // Num cores
				numCores = atoi( optarg );
				break;
		}
	}

	cout << "on node " << myNode << ", numNodes = " << numNodes << ", numCores = " << numCores << endl;

	allocQs();
	launchThreads( numNodes, numCores, myNode );


	MPI_Finalize();
	return 0;
}

