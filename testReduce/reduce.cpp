/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include <stdlib.h>
#include <vector>
#include <iostream>
#include <cassert>
#include <mpi.h>

using namespace std;

class Element
{
	public:
		Element()
		{;}

		vector< unsigned int > values;
		unsigned int maxReduce( unsigned int begin, unsigned int end);

		unsigned int nodeMax;
};

Element e;
pthread_mutex_t mutex;
unsigned int threadsRemaining;
unsigned int numLoops;
unsigned int numThreads;
int numNodes;
int myNode;
unsigned int numObjects;

void threadReduce( void( *func )( unsigned int, bool ), unsigned int );
unsigned int nodeReduce( 
	unsigned int ( *func )( unsigned int*, unsigned int ), unsigned int );

unsigned int getMaxOnNode( unsigned int *v, unsigned int num )
{
	int max = 0;
	for ( unsigned int i = 0; i < num; ++i ) {
		if ( max < v[i] )
			max = v[i];
	}
	return max;
}

void getMaxOnThread( unsigned int v, bool isLast )
{
	static int max = 0;

	if ( v > max ) max = v;

	if ( isLast ) {
		unsigned int ret = nodeReduce( &getMaxOnNode, max );
		cout << myNode << ": max= " << ret << endl;
		max = 0;
	}
}

// Thread safe, runs on all threads with different ranges.
unsigned int Element::maxReduce( unsigned int begin, unsigned int end
	)
{
	assert ( begin < numObjects && end <= numObjects );
	unsigned int max = 0;
	for ( unsigned int i = begin; i < end; ++i ) {
		if ( max < values[i] )
			max = values[i];
	}
	int status = pthread_mutex_lock( &mutex );
		assert( status == 0 );
		if ( nodeMax < max )
			nodeMax = max;
	pthread_mutex_unlock( &mutex );
}

unsigned int nodeReduce(
	unsigned int( *func )( unsigned int*, unsigned int ), unsigned int v )
{
	unsigned int *recvBuf = new unsigned int[ numNodes ];
	MPI_Allgather( &v, 1, MPI_UNSIGNED, recvBuf, numNodes, MPI_UNSIGNED, MPI_COMM_WORLD );
	unsigned int ret = ( *func )( recvBuf, numNodes );
	delete[] recvBuf;
	return ret;
}

void* threadFunc( void* info )
{
	unsigned int myThread = reinterpret_cast< unsigned long >( info );
	unsigned int start = ( numObjects * myThread ) / numThreads;
	unsigned int end = ( numObjects * ( myThread + 1 ) ) / numThreads;

	e.maxReduce( start, end );
}

int main( int argc, char** argv )
{
	if ( argc != 3 ) {
		cout << "Usage: " << argv[0] << " numObjects numThreads\n";
		exit( 0 );
	}

	numObjects = atoi( argv[1] );
	threadsRemaining = numThreads = atoi( argv[2] );
	int status = pthread_mutex_init( &mutex, NULL );
	assert( status == 0 );

	e.values.resize( numObjects );
	e.nodeMax = 0;

	int provided;
	MPI_Init_thread( &argc, &argv, MPI_THREAD_SERIALIZED, &provided );
	MPI_Comm_size( MPI_COMM_WORLD, &numNodes );
	MPI_Comm_rank( MPI_COMM_WORLD, &myNode );

	srandom( 12345 + myNode );
	for ( unsigned int i = 0; i < numObjects; ++i )
		e.values[i] = random();

	pthread_t* threads = new pthread_t[ numThreads ];
	for( unsigned int i = 0; i < numThreads; ++i ) {
		int rc = pthread_create( threads + i, NULL, threadFunc, (void*)i );
		assert( rc == 0 );
	}

	for( unsigned int i = 0; i < numThreads; ++i ) {
		void* status;
		int ret = pthread_join( threads[i], &status );
		if ( ret )
			cout << "Error: Unable to join threads\n";
	}
	delete[] threads;
	pthread_mutex_destroy( &mutex );

	unsigned int ret = nodeReduce( &getMaxOnNode, e.nodeMax );
	cout << myNode << ": max= " << ret << endl;

	MPI_Finalize();
}
