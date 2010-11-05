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

#define QSIZE maxNodes * maxThreads
#define NLOOP 100

using namespace std;

static char* inQ;
static char* outQ;
static char* mpiInQ;
static char* mpiRecvQ;
static int pos[maxThreads]; // Count # of entries on outQ on this thread
static int offset[maxThreads]; // Offset in buffer for this thread.
static bool blockingParserCall;

void allocQs()
{
	inQ = reinterpret_cast< char* >( new Tracker[ QSIZE ] );
	outQ = reinterpret_cast< char* >( new Tracker[ QSIZE ] );
	mpiInQ = reinterpret_cast< char* >( new Tracker[ QSIZE ] );
	mpiRecvQ = reinterpret_cast< char* >( new Tracker[ QSIZE ] );
	for ( int i = 0; i < maxThreads; ++i ) {
		pos[i] = 0;
		offset[i] = ( i * QSIZE ) / maxThreads;
	}
	blockingParserCall = 0;
}

void process( const ProcInfo* p )
{
	;
}

/**
 * Puts data onto queue in location determined by current thread.
 * Has to be this way because otherwise we'll end up with memory clashes.
 */
void addToOutQ( const ProcInfo* p, const Tracker* t )
{
	Tracker* newt = reinterpret_cast< Tracker* >( outQ );
	int i = p->threadIndexInGroup;
	newt[ pos[ i ] + offset[ i ] ] = *t;
	newt[ pos[ i ] + offset[ i ] ].setStop( 0 );
	++pos[i];
}

/**
 * Must be protected by mutex.
 */
void swapQ()
{
	char* temp = inQ;
	inQ = outQ;
	outQ = temp;
	Tracker* t = reinterpret_cast< Tracker* >( inQ );
	for ( int i = 0; i < maxThreads; ++i ) {
		t[ pos[i] + offset[i] ].setStop( 1 );
		pos[i] = 0;
	}
}

/**
 * Must be protected by mutex
 */
void swapMpiQ()
{
	char* temp = mpiInQ;
	mpiInQ = mpiRecvQ;
	mpiRecvQ = temp;
}

/**
 * Reads the specified q and processes contents.
 * Like the main MOOSE system, each thread scans everything and only 
 * executes stuff if the node and thread match.
 * In main moose the pain of this is lessened because this will go through
 * a msg, and each queue entry may have many targets distributed between
 * nodes and threads.
 */
void exec( const ProcInfo* p, const char* q )
{
	const Tracker* t = reinterpret_cast< const Tracker* >( q );
	for ( unsigned int i = 0; i < p->numThreadsInGroup; ++i ) {
		for ( const Tracker* j = t + offset[i]; j->stop() != 1; ++j ) {
			if ( j->node() == static_cast< int >( p->myNode ) &&
				j->thread() == static_cast< int >( p->threadIndexInGroup ) )
			{
				Tracker k = *j;
				k.setNextHop();
				k.print();
				addToOutQ( p, &k );
			}
		}
	}
}

void* eventLoop( void* info )
{
	ProcInfo *p = reinterpret_cast< ProcInfo* >( info );
	cout << "eventLoop on " << p->myNode << ":" << 
		p->threadIndexInGroup << endl;

	int rc;

	for( unsigned int i = 0; i < NLOOP; ++i ) {
		// Phase 1
		process( p );
		// This custom barrier also carries out the swap operation 
		// internally.
		p->barrier1->wait();

		// Phase 2. Here we clean up all the local node Msgs.
		// In parallel, the MPI data transfer begins by broadcasting
		// contents of inQ on node 0.
		exec( p, inQ );

		// Phase 3
		// The allgather approach is not going to scale well: 
		// For N nodes mpiQ needs to set aside N*sizeof(inQ). 
		// Instead, do N bcast calls and interleave with the processing
		// for the data received on the previous bcast call.
		// If we can permit slower internode data transfer then the #
		// of bcast calls goes down.
		for ( unsigned int j = 0; j < p->numNodes; ++j ) {
			p->barrier2->wait(); // This barrier swaps mpiInQ and mpiOutQ
			if ( j != p->myNode )
				exec( p, mpiInQ );
		}

		// Here we use the stock pthreads barrier, whose performance is
		// pretty dismal. Worth comparing with the Butenhof barrier. I
		// earlier wrote a nasty barrier that does a busy-loop but was
		// _much_ faster.
		rc = pthread_barrier_wait( p->barrier3 );
		assert( rc == 0 || rc == PTHREAD_BARRIER_SERIAL_THREAD );
	}
	pthread_exit( NULL );
}



/*
 * Happens on the one thread doing MPI stuff.
 */
void* mpiEventLoop( void* info )
{
	ProcInfo *p = reinterpret_cast< ProcInfo* >( info );
	cout << "mpiEventLoop on " << p->myNode << ":" << 
		p->threadIndexInGroup << endl;

	for( unsigned int i = 0; i < NLOOP; ++i ) {
		// Phase 1: do nothing. But we must wait for barrier 0 to clear,
		// because we need inQ to be settled before broadcasting it.
		p->barrier1->wait();

		// Phase 2, 3. Now we loop around barrier 2 till all nodes have
		// sent data and the data has been received and processed.
		// On the process threads the inQ/mpiInQ is busy being executed.
		for ( unsigned int j = 0; j < p->numNodes; ++j ) {
			if ( p->myNode == j )
				MPI_Bcast( inQ, QSIZE * sizeof( Tracker ), MPI_CHAR, j, MPI_COMM_WORLD );
			else 
				MPI_Bcast( mpiRecvQ, QSIZE * sizeof( Tracker ), MPI_CHAR, j, MPI_COMM_WORLD );
			p->barrier2->wait(); // This barrier swaps mpiInQ and mpiRecvQ
		}

		// Phase 3: Read and execute the arrived MPI data on all threads 
		// except the one which just sent it out.
		// On this thread, we just wait till the final barrier.
		int rc = pthread_barrier_wait( p->barrier3 );
		assert( rc == 0 || rc == PTHREAD_BARRIER_SERIAL_THREAD );
	}
	pthread_exit( NULL );
}

bool isAckPending()
{
	return 0;
}

/*
 * Happens on the one thread doing Shell stuff.
 */
void* shellEventLoop( void* info )
{
	ProcInfo *p = reinterpret_cast< ProcInfo* >( info );
	cout << "shellEventLoop on " << p->myNode << ":" << 
		p->threadIndexInGroup << endl;

	for( unsigned int i = 0; i < NLOOP; ++i ) {
		// Phase 1: Protect the barrier (actually, the swap call)
		// with a mutex so that the Shell doesn't insert data into outQ
		// while things are changing. Note that this outQ is in the
		// Shell group and thus is safe from the other threads.
		pthread_mutex_lock( p->shellSendMutex );
			p->barrier1->wait();
			if ( blockingParserCall && !isAckPending() ) {
				pthread_cond_signal( p->parserBlockCond );
			}
		pthread_mutex_unlock( p->shellSendMutex );

		// Phase 2, 3. Here we simply ignore barriers 2 and 3 as they
		// do not matter for the Shell. This takes a little
		// care when initializing the threads, but saves time.
	}
	pthread_exit( NULL );
}
