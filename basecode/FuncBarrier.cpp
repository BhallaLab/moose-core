/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/
#include <pthread.h>
#include <cassert>
#include "FuncBarrier.h"

/// Creation is initialization.
FuncBarrier::FuncBarrier( unsigned int numThreads, void (*op)() )
	: numThreads_( numThreads ), 
	threadsRemaining_( numThreads ), 
	numLoops_( 0 ), 
	op_( op )
{
	int status = pthread_mutex_init( &mutex_, NULL );
	assert( status == 0 );
	status = pthread_cond_init( &cond_, NULL );
	assert( status == 0 );
}

/// Destruction is cleanup and release
FuncBarrier::~FuncBarrier()
{
	assert ( numThreads_ == threadsRemaining_ );
	int status = pthread_mutex_destroy( &mutex_ );
	assert( status == 0 );
	status = pthread_cond_destroy( &cond_ );
	assert( status == 0 );
}

void FuncBarrier::dummyOp()
{;}

void FuncBarrier::wait()
{
	if ( numThreads_ <= 1 ) { // Bypass all the fancy stuff.
		(*op_)();	// Execute the function here within the mutex
		return;
	}
	int status = pthread_mutex_lock( &mutex_ );
	assert( status == 0 );
	unsigned int loop = numLoops_;
	
	if ( threadsRemaining_ == 1 ) {
		++numLoops_;
		threadsRemaining_ = numThreads_;

		/** 
		 * Execute the function here within the mutex, 
		 * before we issue the all-clear.
		 */
		(*op_)();	

		status = pthread_cond_broadcast( &cond_ );
		assert( status == 0 );
	} else {
		--threadsRemaining_;
		int cancel;
		pthread_setcancelstate( PTHREAD_CANCEL_DISABLE, &cancel );
		while ( loop == numLoops_ ) {
			status = pthread_cond_wait( &cond_, &mutex_ );
			assert( status == 0 );
		}
		int dummy;
		pthread_setcancelstate( cancel, &dummy );
	}
	pthread_mutex_unlock( &mutex_ );
}
