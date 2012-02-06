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

/**
 * processEventLoop
 * Executes all the Process operations, and executes all the recieved msgs.
 * This executes on all the processThreads simultaneously.
 */
void* processEventLoop( void* info )
{
	ProcInfo *p = reinterpret_cast< ProcInfo* >( info );
	// cout << "eventLoop on " << p->nodeIndexInGroup << ":" << p->threadIndexInGroup << endl;
	Clock* clock = reinterpret_cast< Clock* >( Id(1).eref().data() );

	while( Shell::keepLooping() )
	// for( unsigned int i = 0; i < NLOOP; ++i )
	{
		/////////////////////////////////////////////////////////////////
		// Phase 1. Carry out Process calculations on all simulated objects
		/////////////////////////////////////////////////////////////////
		clock->processPhase1( p );
		// This custom barrier carries out the swap operation 
		p->barrier1->wait(); // Within this func, inQ and outQ are swapped.

		/////////////////////////////////////////////////////////////////
		// Phase 2: Execute inQ for all nodes. Transfer data as needed
		/////////////////////////////////////////////////////////////////
		clock->processPhase2( p ); // Do tick juggling for the clock.
		if ( Shell::numNodes() <= 1 ) {
			Qinfo::readQ( p->threadIndexInGroup ); //Deliver all local Msgs
		} else {
			for ( unsigned int j = 0; j < Shell::numNodes(); ++j ) {
#ifdef USE_MPI
				p->barrier2->wait(); // Wait for MPI thread to recv data
#endif
				Qinfo::readQ( p->threadIndexInGroup ); //Deliver all Msgs
			}
		}
		/////////////////////////////////////////////////////////////////
		// Phase 3: Just a barrier to tie things up and do clock scheduling
		/////////////////////////////////////////////////////////////////
		p->barrier3->wait();
	}
	pthread_exit( NULL );
	return 0; //To keep compiler happy.
}

/**
 * mpiEventLoop.
 * Happens on the one thread doing MPI stuff. This variant does an
 * MPI_Bcast to every node in the group, going through each node in turn.
 * The idea is to do calculations on recieved data for each node 
 * at the same time as data for the next node in sequence is transferred.
 */
void* mpiEventLoop( void* info )
{
	ProcInfo *p = reinterpret_cast< ProcInfo* >( info );

	while( Shell::keepLooping() )
	{
		/////////////////////////////////////////////////////////////////
		// Phase 1: do nothing. But we must wait for barrier 0 to clear,
		// because we need inQ to be settled before broadcasting it.
		/////////////////////////////////////////////////////////////////
		p->barrier1->wait(); // Here the inQ is set to the local Q.

		/////////////////////////////////////////////////////////////////
		// Phase 2: Send data, then juggle Queue buffers in the barrier
		/////////////////////////////////////////////////////////////////
#ifdef USE_MPI
		unsigned int actualSize = 0;
		for ( unsigned int j = 0; j < Shell::numNodes(); ++j ) {
			if ( p->nodeIndexInGroup == j ) { // Send out data
				assert( Qinfo::sendQ()[0] >= 2 );
				MPI_Bcast( Qinfo::sendQ(), Qinfo::blockSize(j), 
					MPI_DOUBLE, j, MPI_COMM_WORLD );
				actualSize = Qinfo::sendQ()[0];
				if ( actualSize > Qinfo::blockSize(j) )
					MPI_Bcast( Qinfo::sendQ(), actualSize, 
						MPI_DOUBLE, j, MPI_COMM_WORLD );
			} else { // Receive data
				MPI_Bcast( Qinfo::mpiRecvQ(), Qinfo::blockSize(j), 
					MPI_DOUBLE, j, MPI_COMM_WORLD );
				actualSize = Qinfo::mpiRecvQ()[0];
				assert( actualSize >= 2 );
				if ( actualSize > Qinfo::blockSize(j) ) {
					Qinfo::expandMpiRecvQ( actualSize );
					MPI_Bcast( Qinfo::mpiRecvQ(), actualSize, 
						MPI_DOUBLE, j, MPI_COMM_WORLD );
				}
			}
			// cout << Shell::myNode() << ":" << p->nodeIndexInGroup << ", data comes from node: " << j << ", blockSize = " << Qinfo::blockSize( j ) << ", actualSize= " << actualSize << "\n";

			Qinfo::setSourceNode( j ); // needed for queue juggling.
			p->barrier2->wait(); // This barrier swaps inQ and mpiRecvQ
		}
#endif
		/////////////////////////////////////////////////////////////////
		// Phase 3: Do nothing, just let the Process threads wrap up.
		/////////////////////////////////////////////////////////////////
		p->barrier3->wait();
	}
	pthread_exit( NULL );
	return 0; //To keep compiler happy.
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
			int rc = pthread_create( threads_ + i, NULL, processEventLoop, 
				(void *)&p[i] );
			assert( rc == 0 );
		} else if ( numNodes_ > 1 && i == numProcessThreads_ ) { // mpiThread stufff.
			int rc = pthread_create( 
				threads_ + i, NULL, mpiEventLoop, (void *)&p[i] );
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

