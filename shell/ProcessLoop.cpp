/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

// #include <unistd.h>
#include <pthread.h>
#ifdef USE_MPI
#include <mpi.h>
#endif
#include "FuncBarrier.h"

// #include <stdlib.h>
#include "header.h"
#include "../scheduling/Tick.h"
#include "../scheduling/TickMgr.h"
#include "../scheduling/TickPtr.h"
#include "../scheduling/Clock.h"

#include "Shell.h"

static const unsigned int BLOCKSIZE = 20000; // duplicate defn, other in Qinfo.cpp

/*
void Shell::eventLoopSingleThreaded()
{
	while ( Shell::keepLooping() )
	{
		clock->processPhase1( &p_ ); // Call Process on all scheduled Objs.
		Qinfo::swapQ();				// Queue swap
		clock->processPhase2( p ); // Do tick juggling for the clock.
		Qinfo::readQ( p->threadIndexInGroup ); //Deliver all local node Msgs
		Clock::checkProcState();	// Decide whether to continue.
	}
}
*/

void* eventLoopForBcast( void* info )
{
	ProcInfo *p = reinterpret_cast< ProcInfo* >( info );
	// cout << "eventLoop on " << p->nodeIndexInGroup << ":" << p->threadIndexInGroup << endl;
	Clock* clock = reinterpret_cast< Clock* >( Id(1).eref().data() );

	while( Shell::keepLooping() )
	// for( unsigned int i = 0; i < NLOOP; ++i )
	{
		// Phase 1. Here we carry out the Process calculations
		// on all simulated objects.
		clock->processPhase1( p );
		// This custom barrier also carries out the swap operation 
		// internally.
		// cout << Shell::myNode() << ":" << p->threadIndexInGroup << ":	phase1 :	" << loopNum << "\n";
		p->barrier1->wait(); // Within this func, inQ and outQ are swapped.

		// Phase 2.
		// Then we clean up all the local node Msgs.
		// In parallel, the MPI data transfer begins by broadcasting
		// contents of inQ on node 0.
		clock->processPhase2( p ); // Do tick juggling for the clock.
		Qinfo::readQ( p->threadIndexInGroup ); //Deliver all local node Msgs

		// Phase 3
		// The allgather approach is not going to scale well: 
		// For N nodes mpiQ needs to set aside N*sizeof(inQ). 
		// Instead, do N bcast calls and interleave with the processing
		// for the data received on the previous bcast call.
		// If we can permit slower internode data transfer then the #
		// of bcast calls goes down.

		if ( Shell::numNodes() > 1 ) {
			for ( unsigned int j = 0; j < Shell::numNodes(); ++j )
				{
					// cout << Shell::myNode() << ":" << p->nodeIndexInGroup << "	: eventLoop, (numSimGroup, tgtNode) = (" << i << ", " << j << ")\n";
					p->barrier2->wait(); // This barrier swaps mpiInQ and mpiRecvQ
					// This internally checks if we are within the allowed
					// blocksize for this buffer. If not, it pushes up a call
					// to the Clock to schedule another cycle to finish of the
					// data transfer.
					Qinfo::readMpiQ( p->threadIndexInGroup, j ); 
				}
		}
		// cout << Shell::myNode() << ":" << p->threadIndexInGroup << ":	phase3 :	" << loopNum++ << "\n";

		// This barrier handles the state transitions for clock scheduling
		// as its internal protected function.
		p->barrier3->wait();
	}
	pthread_exit( NULL );
}

/*
 * Happens on the one thread doing MPI stuff. This variant does an
 * MPI_Bcast to every node in the group. 
 */
void* mpiEventLoopForBcast( void* info )
{
	ProcInfo *p = reinterpret_cast< ProcInfo* >( info );
	// cout << "mpiEventLoop on " << p->nodeIndexInGroup << ":" << p->threadIndexInGroup << endl;

	while( Shell::keepLooping() )
	{
		// Phase 1: do nothing. But we must wait for barrier 0 to clear,
		// because we need inQ to be settled before broadcasting it.
		p->barrier1->wait();

		// Phase 2, 3. Now we loop around barrier 2 till all nodes have
		// sent data and the data has been received and processed.
		// On the process threads the inQ/mpiInQ is busy being executed.
		for ( unsigned int j = 0; j < Shell::numNodes(); ++j )
		{
#ifdef USE_MPI
				// cout << Shell::myNode() << ":" << p->nodeIndexInGroup << "	: mpiEventLoop, (numSimGroup, tgtNode) = (" << i << ", " << j << ")\n";
				if ( p->nodeIndexInGroup == j ) {
					MPI_Bcast( Qinfo::inQ( i ), BLOCKSIZE, MPI_CHAR, j, 
						MPI_COMM_WORLD );
					unsigned int actualSize = 
						*reinterpret_cast< unsigned int* >( Qinfo::inQ(i) );
					if ( actualSize > BLOCKSIZE )
						MPI_Bcast( Qinfo::inQ( i ), actualSize, MPI_CHAR, j,
							MPI_COMM_WORLD );
				// cout << Shell::myNode() << ":" << p->nodeIndexInGroup << "	: mpiEventLoop outgoing, actualSize= " << actualSize << ", DestNode=" << j << ")\n";
				} else {
					MPI_Bcast( Qinfo::mpiRecvQbuf(), BLOCKSIZE, MPI_CHAR, j,
						MPI_COMM_WORLD );
					unsigned int actualSize = 
						*reinterpret_cast< unsigned int* >(
						Qinfo::mpiRecvQbuf() );
					if ( actualSize > BLOCKSIZE ) {
						Qinfo::expandMpiRecvQbuf( actualSize );
						MPI_Bcast( Qinfo::mpiRecvQbuf(), actualSize, 
							MPI_CHAR, j, MPI_COMM_WORLD );
					}
				// cout << Shell::myNode() << ":" << p->nodeIndexInGroup << "	: mpiEventLoop incoming, actualSize= " << actualSize << ", DestNode=" << j << ")\n";
				}

#endif
				p->barrier2->wait(); // This barrier swaps mpiInQ and mpiRecvQ
		}

		// Phase 3: Read and execute the arrived MPI data on all threads 
		// except the one which just sent it out.
		// On this thread, we just wait till the final barrier.
		p->barrier3->wait();
		// int rc = pthread_barrier_wait( p->barrier3 );
		// assert( rc == 0 || rc == PTHREAD_BARRIER_SERIAL_THREAD );
	}
	pthread_exit( NULL );
}

void* reportGraphics( void* info )
{
	// ProcInfo *p = reinterpret_cast< ProcInfo* >( info );
	// cout << "reportGraphics on " << p->nodeIndexInGroup << ":" << p->threadIndexInGroup << endl;
	pthread_exit( NULL );
}

//////////////////////////////////////////////////////////////////////////
// This function sets up the threading for the process loop.
//////////////////////////////////////////////////////////////////////////
void Shell::launchThreads()
{
	attr_ = new pthread_attr_t;
	pthread_attr_init( attr_ );
	pthread_attr_setdetachstate( attr_, PTHREAD_CREATE_JOINABLE );

	// Add one for the MPI thread if we have multiple nodes.
	unsigned int numThreads = numProcessThreads_ + ( numNodes_ > 1 ); 

	barrier1_ = new FuncBarrier( numThreads, &Qinfo::swapQ );
	barrier2_ = new FuncBarrier( numThreads, &Qinfo::swapMpiQ );
	barrier3_ = new FuncBarrier( numThreads, &Clock::checkProcState );
	int ret;

	parserMutex_ = new pthread_mutex_t; // Assign the Shell variables.
	parserBlockCond_ = new pthread_cond_t;

	ret = pthread_mutex_init( parserMutex_, NULL );
	assert( ret == 0 );

	ret = pthread_cond_init( parserBlockCond_, NULL );
	assert( ret == 0 );

	keepLooping_ = 1;
	
	threadProcs_.resize( numThreads );
	vector< ProcInfo >& p = threadProcs_;
	// An extra thread is used by MPI, and on node 0, yet another for Shell
	// pthread_t* threads = new pthread_t[ numThreads ];
	threads_ = new pthread_t[ numThreads ];

	for ( unsigned int i = 0; i < numThreads; ++i ) {
		// Note that here we put # of compute cores, not total threads.
		p[i].numThreadsInGroup = numProcessThreads_; 
		p[i].groupId = 1; // Later more sophisticated subdivision into grps
		p[i].threadIndexInGroup = i + 1;
		p[i].nodeIndexInGroup = myNode_;
		p[i].numNodesInGroup = numNodes_;
		p[i].barrier1 = barrier1_;
		p[i].barrier2 = barrier2_;
		p[i].barrier3 = barrier3_;
		p[i].procIndex = i;

	// cout << myNode_ << "." << i << ": ptr= " << &( p[i] ) << ", Shell::procInfo = " << &p_ << " setting up procs\n";
		if ( i < numProcessThreads_ ) { // These are the compute threads
			int rc = pthread_create( threads_ + i, NULL, eventLoopForBcast, 
				(void *)&p[i] );
			assert( rc == 0 );
		} else if ( numNodes_ > 1 && i == numProcessThreads_ ) { // mpiThread stufff.
			int rc = pthread_create( 
				threads_ + i, NULL, mpiEventLoopForBcast, (void *)&p[i] );
			assert( rc == 0 );
		}
	}
}

void Shell::joinThreads()
{
	// Add one for the MPI thread if needed.
	int numThreads = numProcessThreads_ + ( numNodes_ > 1 ); 
	int ret;

	for ( int i = 0; i < numThreads; ++i ) {
		void* status;
		ret = pthread_join( threads_[i], &status );
		if ( ret )
			cout << "Error: Unable to join threads\n";
	}

	delete[] threads_;
	pthread_attr_destroy( attr_ );
	delete attr_;
	ret = pthread_mutex_destroy( parserMutex_ );
	delete parserMutex_;
	ret = pthread_cond_destroy( parserBlockCond_ );
	delete parserBlockCond_;

	delete barrier1_;
	delete barrier2_;
	delete barrier3_;

	assert( ret == 0 );
}

