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
#include "ProcInfo.h"
#include "Tracker.h"

#define QSIZE 256

using namespace std;

static char* inQ;
static char* outQ;
static char* mpiQ;
static int pos[maxThreads]; // Count # of entries on outQ on this thread
static int offset[maxThreads]; // Offset in buffer for this thread.

void addToOutQ( const ProcInfo* p, const Tracker* t );

void allocQs()
{
	inQ = reinterpret_cast< char* >( new Tracker[ QSIZE ] );
	outQ = reinterpret_cast< char* >( new Tracker[ QSIZE ] );
	mpiQ = reinterpret_cast< char* >( new Tracker[ QSIZE ] );
	for ( int i = 0; i < maxThreads; ++i ) {
		pos[i] = 0;
		offset[i] = ( i * QSIZE ) / maxThreads;
	}

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

	for( unsigned int i = 0; i < 10; ++i ) {
		// Phase 1
		process( p );
		swapQ();
		int rc = pthread_barrier_wait( p->barrier1 );
		assert( rc == 0 || rc == PTHREAD_BARRIER_SERIAL_THREAD );

		// Phase 2
		exec( p, inQ );
		rc = pthread_barrier_wait( p->barrier2 );
		assert( rc == 0 || rc == PTHREAD_BARRIER_SERIAL_THREAD );

		// Phase 3
		exec( p, mpiQ );
		rc = pthread_barrier_wait( p->barrier3 );
		assert( rc == 0 || rc == PTHREAD_BARRIER_SERIAL_THREAD );
	}
	pthread_exit( NULL );
}



/*
void* mpiEventLoop( void* info )
{
	ProcInfo *p = reinterpret_cast< ProcInfo* >( info );
	cout << "mpiEventLoop on " << p->myNode << ":" << 
		p->threadIndexInGroup << endl;

	for( unsigned int i = 0; i < 100; ++i ) {
		// Phase 1: do nothing
		int rc = pthread_barrier_wait( p->barrier1 );
		assert( rc == 0 || rc == PTHREAD_BARRIER_SERIAL_THREAD );

		// Phase 2: Exchange MPI data. InQ has just become available.
		mpiSend( p, inQ, mpiQ );
		rc = pthread_barrier_wait( p->barrier2 );
		assert( rc == 0 || rc == PTHREAD_BARRIER_SERIAL_THREAD );

		// Phase 3: Do nothing.
		rc = pthread_barrier_wait( p->barrier3 );
		assert( rc == 0 || rc == PTHREAD_BARRIER_SERIAL_THREAD );
	}
	pthread_exit( NULL );
}

void* shellEventLoop( void* info )
{
	ProcInfo *p = reinterpret_cast< ProcInfo* >( info );
	cout << "mpiEventLoop on " << p->myNode << ":" << 
		p->threadIndexInGroup << endl;

	for( unsigned int i = 0; i < 100; ++i ) {
		// Phase 1: Data from parser into server loop
		// Mutex lock. 
		// Check flag for data arrival, if so, put into
		// Q for group 0 which deals with off-node stuff. 
		// Mutex unlock
		int rc = pthread_barrier_wait( p->barrier1 );
		assert( rc == 0 || rc == PTHREAD_BARRIER_SERIAL_THREAD );

		// Phase 2: Data from server to parser
		// Mutex lock
		// Copy the InQ for Group0 over to parser thread, or something.

		// Phase 2: do nothing.
		mpiSend( p, inQ, mpiQ );
		rc = pthread_barrier_wait( p->barrier2 );
		assert( rc == 0 || rc == PTHREAD_BARRIER_SERIAL_THREAD );

		// Phase 3: Do nothing.
		rc = pthread_barrier_wait( p->barrier3 );
		assert( rc == 0 || rc == PTHREAD_BARRIER_SERIAL_THREAD );
	}
	pthread_exit( NULL );
}
*/
