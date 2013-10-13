/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

// #include <unistd.h>
#ifdef USE_MPI
#include <mpi.h>
#endif

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
		Qinfo::swapQ();

		/////////////////////////////////////////////////////////////////
		// Phase 2: Execute inQ for all nodes. Transfer data as needed
		/////////////////////////////////////////////////////////////////
		clock->processPhase2( p ); // Do tick juggling for the clock.
		if ( Shell::numNodes() <= 1 ) {
			Qinfo::readQ( p->threadIndexInGroup ); //Deliver all local Msgs
		} else {
			for ( unsigned int j = 0; j < Shell::numNodes(); ++j ) {
#ifdef USE_MPI
				Qinfo::swapMpiQ;
#endif
				Qinfo::readQ( p->threadIndexInGroup ); //Deliver all Msgs
			}
		}
		/////////////////////////////////////////////////////////////////
		// Phase 3: Just a barrier to tie things up and do clock scheduling
		/////////////////////////////////////////////////////////////////
		Clock::checkProcState();
	}
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
	while( Shell::keepLooping() )
	{
		/////////////////////////////////////////////////////////////////
		// Phase 1: do nothing. But we must wait for barrier 0 to clear,
		// because we need inQ to be settled before broadcasting it.
		/////////////////////////////////////////////////////////////////
		Qinfo::swapQ();

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
			Qinfo::swapMpiQ();
		}
#endif
		/////////////////////////////////////////////////////////////////
		// Phase 3: Do nothing, just let the Process threads wrap up.
		/////////////////////////////////////////////////////////////////
		Clock::checkProcState();
	}
	return 0; //To keep compiler happy.
}

//////////////////////////////////////////////////////////////////////////
// This function sets up the threading for the process loop.
//////////////////////////////////////////////////////////////////////////
void Shell::launchThreads()
{
	// Add one for the MPI thread if we have multiple nodes.
	unsigned int numThreads = numProcessThreads_ + ( numNodes_ > 1 ); 
	keepLooping_ = 1;
	threadProcs_.resize( numThreads );
	vector< ProcInfo >& p = threadProcs_;
	// Have to prevent the parser thread from doing stuff during 
	// the process loop, except at very tightly controlled times.
	for ( unsigned int i = 0; i < numThreads; ++i ) {
		// Note that here we put # of compute cores, not total threads.
		p[i].numThreadsInGroup = numProcessThreads_; 
		p[i].groupId = 1; // Later more sophisticated subdivision into grps
		p[i].threadIndexInGroup = i + 1;
		p[i].nodeIndexInGroup = myNode_;
		p[i].numNodesInGroup = numNodes_;
		p[i].procIndex = i;

	// cout << myNode_ << "." << i << ": ptr= " << &( p[i] ) << ", Shell::procInfo = " << &p_ << " setting up procs\n";
	}
}
