/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include <iostream>
#include <vector>
#include <unistd.h>
#include <mpi.h>
#include <pthread.h>
#include <cassert>
#include <stdlib.h>
#include "FuncBarrier.h"
#include "ProcInfo.h"
#include "Tracker.h"

using namespace std;
void addToOutQ( const ProcInfo* p, const Tracker* t );
void* eventLoop( void* info );
void* mpiEventLoop( void* info );
void allocQs();
void swapQ();
void swapMpiQ();

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
	// pthread_barrier_t barrier1;
	// pthread_barrier_t barrier2;
	FuncBarrier barrier1( numThreads, &swapQ );
	FuncBarrier barrier2( numThreads, &swapMpiQ );
	pthread_barrier_t barrier3;
	int ret;
	/*
	ret = pthread_barrier_init( &barrier1, NULL, numCores );
	assert( ret == 0 );
	ret = pthread_barrier_init( &barrier2, NULL, numCores );
	assert( ret == 0 );
	*/
	ret = pthread_barrier_init( &barrier3, NULL, numThreads );
	assert( ret == 0 );

	pthread_t gThread;
	if ( myNode == 0 ) { // Launch graphics thread only on node 0.
		ProcInfo p;
		// pthread_barrier_t barrier1;
		int rc = pthread_create(&gThread, NULL, reportGraphics, 
			(void *)&p );
		if ( rc )
			cout << "Error: return code from pthread_create: " << rc << endl;
	}

	vector< ProcInfo > p( numCores + 1 );
	// An extra thread is used by MPI.
	pthread_t* threads = new pthread_t[ numCores + 1 ];

	for ( int i = 0; i < numThreads; ++i ) {
		// Note that here we put # of compute cores, not total threads.
		p[i].numThreadsInGroup = numCores; 

		p[i].threadIndexInGroup = i;
		p[i].myNode = myNode;
		p[i].numNodes = numNodes;
		p[i].barrier1 = &barrier1;
		p[i].barrier2 = &barrier2;
		p[i].barrier3 = &barrier3;

		if ( i < numCores ) {
			if ( myNode == 0 && i == 0 ) { // For now just set off rule 0
				Tracker t( numNodes, numCores, Rule( i % 4 ) );
				addToOutQ( &p[i], &t );
			}
			int rc = pthread_create( threads + i, NULL, eventLoop, 
				(void *)&p[i] );
			assert( rc == 0 );
		} else if ( i == numCores ) { // mpiThread stufff.
			int rc = pthread_create( 
				threads + numCores, NULL, mpiEventLoop, (void *)&p[i] );
			assert( rc == 0 );
		}
	}


	// clean up. Add an extra time round loop for the MPI thread.
	for ( int i = 0; i < numThreads; ++i ) {
		void* status;
		int ret = pthread_join( threads[i], &status );
		if ( ret )
			cout << "Error: Unable to join threads\n";
	}

	if ( myNode == 0 ) { // clean up graphics thread only on node 0.
		void* status;
		int ret = pthread_join( gThread, &status );
		if ( ret )
			cout << "Error: Unable to join threads\n";
	}

	delete[] threads;
	pthread_attr_destroy( &attr );
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


/*
loop( void* threadid )
{
	process
	barrier1
	local exec
	barrier2
	offnode exec
	barrier3
}
*/
