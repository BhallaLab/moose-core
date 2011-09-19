/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

// #include <unistd.h>
#include <vector>
#include <iostream>
#include <math.h>
#include <pthread.h>
#include <stdlib.h>
#include <cassert>
#include "FuncBarrier.h"

using namespace std;

static FuncBarrier* barrier;

class ThreadInfo 
{
	public:
		ThreadInfo( int threadNum, int numJoins, int numBusy )
			: threadNum_( threadNum ),
				numJoins_( numJoins ),
				numBusy_( numBusy )

		{;}

		int threadNum() const
		{
			return threadNum_;
		}

		int numJoins() const
		{
			return numJoins_;
		}

		int numBusy() const
		{
			return numBusy_;
		}

	private:
		int threadNum_;
		int numJoins_;
		int numBusy_;
};

void busyFunc( const ThreadInfo* t )
{
	int numBusy = t->numBusy();

	for ( int i = 0; i < numBusy; ++i ) {
		vector< double > temp( i );
		for ( int j = 0; j < i; ++j ) {
			temp[j] = sin( j );
		}
	}
}

/// This is the function that happens in the barrier.
void barrierOp()
{
}

/// This is the function that happens on each thread if it is a sustained
/// thread, using barriers to synchronize.
void* barrierLoop( void* info )
{
	ThreadInfo* t = reinterpret_cast< ThreadInfo* >( info );
	for ( int i = 0; i < t->numJoins(); ++i ) {
		barrier->wait();
		busyFunc( t );
	}
	pthread_exit( NULL );
}

void* joinLoop( void* info )
{
	ThreadInfo* t = reinterpret_cast< ThreadInfo* >( info );
	busyFunc( t );
	pthread_exit( NULL );
}

//////////////////////////////////////////////////////////////////////////
// This function sets up the threading for the process loop.
//////////////////////////////////////////////////////////////////////////
int main( int argc, char** argv )
{
	if ( argc < 5 ) {
		cout << "Usage: " << argv[0] << " numThreads numJoins numBusy doBarrier\n";
		return 1;
	}
	int numThreads = atoi( argv[1] );
	int numJoins = atoi( argv[2] );
	int numBusy = atoi( argv[3] );
	int doBarrier = atoi( argv[4] );

	int ret;

	pthread_attr_t* attr = new pthread_attr_t;
	pthread_attr_init( attr );
	pthread_attr_setdetachstate( attr, PTHREAD_CREATE_JOINABLE );

	pthread_t* threads = new pthread_t[ numThreads ];
	vector< ThreadInfo* > threadInfos( numThreads );

	for ( int i = 0; i < numThreads; ++i )
		threadInfos[i] = new ThreadInfo( i, numJoins, numBusy );

	if ( doBarrier ) {
		barrier = new FuncBarrier( numThreads, &barrierOp );
		/*
		pthread_mutex_t* parserMutex = new pthread_mutex_t; // Assign the Shell variables.
		pthread_cond_t* parserBlockCond = new pthread_cond_t;
	
		ret = pthread_mutex_init( parserMutex, NULL );
		assert( ret == 0 );
	
		ret = pthread_cond_init( parserBlockCond, NULL );
		assert( ret == 0 );
		*/
	
		for ( int i = 0; i < numThreads; ++i ) {
			int rc = pthread_create( threads + i, NULL, barrierLoop, 
				(void *)threadInfos[i] );
			assert( rc == 0 );
		}
	
		for ( int i = 0; i < numThreads; ++i ) {
			void* status;
			ret = pthread_join( threads[i], &status );
			if ( ret )
				cout << "Error: Unable to join threads\n";
		}
	} else {
		for ( int j = 0; j < numJoins; ++j ) {
			for ( int i = 0; i < numThreads; ++i ) {
				int rc = pthread_create( threads + i, NULL, joinLoop, 
					(void *)threadInfos[i] );
				assert( rc == 0 );
			}
	
			for ( int i = 0; i < numThreads; ++i ) {
				void* status;
				ret = pthread_join( threads[i], &status );
				if ( ret )
					cout << "Error: Unable to join threads\n";
			}
		}
	}

	delete[] threads;
	pthread_attr_destroy( attr );
	delete attr;
	/*
	ret = pthread_mutex_destroy( parserMutex );
	delete parserMutex;
	ret = pthread_cond_destroy( parserBlockCond );
	delete parserBlockCond;
	*/

	delete barrier;

	return 0;
}
