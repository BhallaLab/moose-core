/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

/**
 * This file contains the thread and MPI-handling functions in Shell.
 */

#include <pthread.h>
// #include <mpi.h>
#include "header.h"
#include "Shell.h"
#include "Dinfo.h"

// Want to separate out this search path into the Makefile options
#include "../scheduling/Tick.h"
#include "../scheduling/TickMgr.h"
#include "../scheduling/TickPtr.h"
#include "../scheduling/ThreadInfo.h"
#include "../scheduling/Clock.h"

#define USE_NODES 1

/**
 * Initialize acks. This call should be done before the 'send' goes out,
 * because with the wonders of threading we might get a response to the
 * 'send' before this call is executed.
 * This MUST be followed by a waitForAck call.
 */
void Shell::initAck()
{
	numAcks_ = 0;
	if ( !isSingleThreaded_ ) {
		pthread_mutex_lock( 
			reinterpret_cast< pthread_mutex_t * >( parserMutex_ ) );
		isBlockedOnParser_ = 1;
		acked_.assign( numNodes_, 0 );
	}
}

/**
 * test for completion of request. This MUST be preceded by an initAck
 * call.
 */
void Shell::waitForAck()
{
	if ( isSingleThreaded_ ) {
		Qinfo::clearQ( &p_ );
	} else {
		while ( isAckPending() )
			pthread_cond_wait( 
				reinterpret_cast< pthread_cond_t * >( parserBlockCond_ ), 
				reinterpret_cast< pthread_mutex_t * >( parserMutex_ ) );
		isBlockedOnParser_ = 0;
		pthread_mutex_unlock( 
			reinterpret_cast< pthread_mutex_t * >( parserMutex_ ) );
	}
}

void Shell::setRunning( bool value )
{
	isRunning_ = value;
}

/**
 * Launches Parser. Blocking when the parser blocks.
 */
void Shell::launchParser()
{
	Id shellId;
	
	cout << "moose : " << flush;
	while ( !getQuit() ) {
		string temp;
		cin >> temp;
		if ( temp == "quit" || temp == "q" ) {
			doQuit();
		}
	}
	cout << "\nQuitting Moose\n" << flush;
}

// Function to assign hardware availability
void Shell::setHardware( 
	bool isSingleThreaded, unsigned int numCores, unsigned int numNodes,
	unsigned int myNode )
{
	isSingleThreaded_ = isSingleThreaded;
	Qinfo::addSimGroup( 1, numNodes ); // This is the parser thread.
	if ( !isSingleThreaded ) {
		// Create the parser and the gui threads.
		numCores_ = numCores;
		numNodes_ = numNodes;
		// The zero queue is for system calls. Then there is one queue
		// per local thread. Each off-node gets another queue.
		// Note the more complex 'group' orgn for
		// eventual highly multithreaded architectures, discussed in
		// NOTES 10 Dec 2009.
		// Qinfo::setNumQs( numCores_ + numNodes_, 1024 );
		//
		// Create thread for managing MPI. Different MPI implementations
		// have different degrees of thread support, so I'll just put
		// the whole MPI handling loop on one thread. The MPI stuff is all 
		// non-blocking so the thread just goes around checking for message
		// completion, setting flags and dispatching stuff.
		
	} else {
		numCores_ = 1;
		numNodes_ = 1;
		// Qinfo::setNumQs( 1, 1024 );
	}
	myNode_ = myNode;
	p_.numNodesInGroup = numNodes_;
	p_.nodeIndexInGroup = myNode;
	acked_.resize( numNodes, 0 );
}

/**
 * Regular shell function that requires that the information about the
 * hardware have been loaded in. For now the function just assigns SimGroups
 */
void Shell::loadBalance()
{
	// Need more info here on how to set up groups distributed over
	// nodes. In fact this will have to be computed _after_ the
	// simulation is loaded. Will also need quite a bit of juggling between
	// nodes when things get really scaled up.
	//
	// Note that the messages have to be rebuilt after this call.
	// Note that this function is called independently on each node.
	if ( !isSingleThreaded_ ) {
		// for ( unsigned int i = 0; i < numNodes_; ++i )
			//These are the worker threads.
			Qinfo::addSimGroup( numCores_, numNodes_ );
	}
}

unsigned int Shell::numCores()
{
	return numCores_;
}

unsigned int Shell::numNodes()
{
	return numNodes_;
}

unsigned int Shell::myNode()
{
	return myNode_;
}

pthread_mutex_t* Shell::parserMutex() const
{
	return parserMutex_;
}
pthread_cond_t* Shell::parserBlockCond() const
{
	return parserBlockCond_;
}

bool Shell::inBlockingParserCall() const
{
	return isBlockedOnParser_;
}


////////////////////////////////////////////////////////////////////////
// Functions for setting off clocked processes.
////////////////////////////////////////////////////////////////////////

void Shell::start( double runtime )
{
	/* Send msg to Clock
	*/
}
