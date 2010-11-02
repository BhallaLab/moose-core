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

using namespace std;

static char* inQ;
static char* outQ;
static char* mpiQ;

void allocQs()
{
	inQ = new char[ sizeof( Tracker ) * 100 ];
	outQ = new char[ sizeof( Tracker ) * 100 ];
	mpiQ = new char[ sizeof( Tracker ) * 100 ];
}

void process( const ProcInfo* p )
{
}

void exec( const ProcInfo* p, const char* q )
{
}

void* eventLoop( void* info )
{
	ProcInfo *p = reinterpret_cast< ProcInfo* >( info );
	cout << "eventLoop on " << p->myNode << ":" << 
		p->threadIndexInGroup << endl;

	for( unsigned int i = 0; i < 100; ++i ) {
		// Phase 1
		process( p );
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
