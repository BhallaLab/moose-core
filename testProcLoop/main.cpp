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

using namespace std;

class ProcInfo {
	public:
		ProcInfo()
			: 
				threadIndexInGroup( 0 ),
				numThreadsInGroup( 1 ),
				groupId( 0 ),
				myNode( 0 ),
				numNodes( 1 )
		{;}

		unsigned int threadIndexInGroup;
		unsigned int numThreadsInGroup;
		unsigned int groupId;
		unsigned int myNode;
		unsigned int numNodes;

		pthread_barrier_t* barrier1;
		pthread_barrier_t* barrier2;
		pthread_barrier_t* barrier3;
};

void* reportGraphics( void* info )
{
	ProcInfo *p = reinterpret_cast< ProcInfo* >( info );
	cout << "reportGraphics on " << p->myNode << ":" << 
		p->threadIndexInGroup << endl;
	pthread_exit( NULL );
}

void* process( void* info )
{
	ProcInfo *p = reinterpret_cast< ProcInfo* >( info );
	cout << "process on " << p->myNode << ":" << 
		p->threadIndexInGroup << endl;

	for( unsigned int i = 0; i < 100; ++i ) {
		int rc = pthread_barrier_wait( p->barrier1 );
		assert( rc == 0 || rc == PTHREAD_BARRIER_SERIAL_THREAD );
		rc = pthread_barrier_wait( p->barrier2 );
		assert( rc == 0 || rc == PTHREAD_BARRIER_SERIAL_THREAD );
		rc = pthread_barrier_wait( p->barrier3 );
		assert( rc == 0 || rc == PTHREAD_BARRIER_SERIAL_THREAD );
	}
	pthread_exit( NULL );
}

void launchThreads( int numNodes, int numCores, int myNode )
{
	pthread_attr_t attr;
	pthread_attr_init( &attr );
	pthread_attr_setdetachstate( &attr, PTHREAD_CREATE_JOINABLE );
	pthread_barrier_t barrier1;
	pthread_barrier_t barrier2;
	pthread_barrier_t barrier3;
	int ret = pthread_barrier_init( &barrier1, NULL, numCores );
	assert( ret == 0 );
	ret = pthread_barrier_init( &barrier2, NULL, numCores );
	assert( ret == 0 );
	ret = pthread_barrier_init( &barrier3, NULL, numCores );
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

	vector< ProcInfo > p( numCores );
	pthread_t* threads = new pthread_t[ numCores ];

	for ( int i = 0; i < numCores; ++i ) {
		p[i].numThreadsInGroup = numCores;
		p[i].threadIndexInGroup = i;
		p[i].myNode = myNode;
		p[i].barrier1 = &barrier1;
		p[i].barrier2 = &barrier2;
		p[i].barrier3 = &barrier3;
		int rc = pthread_create( threads + i, NULL, process, 
			(void *)&p[i] );
		if ( rc )
			cout << "Error: return code from pthread_create: " << rc << endl;
	}

	// clean up
	for ( int i = 0; i < numCores; ++i ) {
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
