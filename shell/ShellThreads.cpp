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
#include "../scheduling/TickPtr.h"
#include "../scheduling/ThreadInfo.h"
#include "../scheduling/Clock.h"

#define USE_NODES 1

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

/**
 * Sets of message loop either on a separate thread, or the current one.
 */
void Shell::launchMsgLoop( Element* shelle )
{
	msgLoop( shelle );
}

// Static func, passed in to the thread.
// This function handles local node as well as off-node messaging 
// for the Shell. 
void* Shell::msgLoop( void* shellePtr )
{
	Element* shelle = reinterpret_cast< Element* >( shellePtr );
	Shell* shell = reinterpret_cast< Shell* >( shelle->dataHandler()->data( 0 ) );
	while ( !shell->getQuit() ) {
		// Put in a condition_wait here in case we have to sync with the
		// Clock. The idea is that if the clock gets going, this
		// loop must wait.
		// cout << myNode_ << ": in Shell::msgLoop, numNodes=" << numNodes_ <<"\n";
		shell->passThroughMsgQs( shelle );
	}
	// cout << myNode_ << ": Shell::msgLoop, exiting\n";
	// pthread_exit( NULL );
	return 0;
}

// This is a single pass through the Shell messaging
// It can be called as a separate thread or as a serial
// function by the Clock, interleaved between data handled by the 
// processing threads.
void Shell::passThroughMsgQs( Element* shelle )
{
	Qinfo::mpiClearQ( &p_ );
	/*
	Qinfo::mergeQ( 0 ); // Fill up inQ
	p_.numNodesInGroup = numNodes_;
	Qinfo::sendAllToAll( &p_ ); // Send out inQ
	// send alltoall. Could do on another thread.
	Qinfo::readQ( &p_ ); // execute stuff on inQ
	Qinfo::readMpiQ( &p_ ); // execute stuff on mpiQ that came in alltoall
	*/
	shelle->process( &p_ );
}


// Static func, passed in to the thread.
// Version 2, dated 24 March. This variant uses alltoall
void* Shell::mpiThreadFunc( void* shellPtr )
{
	Shell* shell = reinterpret_cast< Shell* >( shellPtr );
	assert( shell->numNodes_ > 1 );
	assert( shell->barrier_ );
	cout << "mpiThreadFunc: on node " << shell->myNode_ << endl;

	while( shell->isRunning_ ) {
		// Sync with first step in Tick::advance
		int rc = pthread_barrier_wait(
			reinterpret_cast< pthread_barrier_t* >( shell->barrier_ ) );
		assert( rc == 0 || rc == PTHREAD_BARRIER_SERIAL_THREAD );
		if ( !shell->isRunning_ )
			break;
			// Now inQ is being updated.
			cout << "mpiThreadFunc: waiting for inQ to fill, clearing mpiQ\n";
			// wait till inQ is filled
		// Now inQ has been filled.
		rc = pthread_barrier_wait(
			reinterpret_cast< pthread_barrier_t* >( shell->barrier_ ) );
		assert( rc == 0 || rc == PTHREAD_BARRIER_SERIAL_THREAD );
		if ( !shell->isRunning_ )
			break;
		// Now inQ is filled and is safe to use.
			cout << "mpiThreadFunc: inQ ready, used by readQ and alltoall\n";

		// start MPI_alltoall
		// barrier till localnode proc is done on all threads in Tick::advance
		// Barrier till localnode cleans up all mpiQ stuff too.
		// Clean up mpiQ for next round of incoming stuff.
		rc = pthread_barrier_wait(
			reinterpret_cast< pthread_barrier_t* >( shell->barrier_ ) );
		assert( rc == 0 || rc == PTHREAD_BARRIER_SERIAL_THREAD );
		cout << "mpiThreadFunc: mpiQ ready, used by readMpiQ\n";
		
	
	}
	
	pthread_exit( NULL );
}

#if 0
// Static func, passed in to the thread.
void Shell::mpiThreadFunc( void* shellPtr )
{
	Shell* shell = reinterpret_cast< Shell* >( shellPtr );



	int numDone = Request::TestSome( numRecvReq_, requestArray_, 
		requestIndices_, requestStatus_ );
	
	for ( int i = 0; i < numDone; ++i ) {
		// Tell relevant SimGroup that this request is done.
	}

	for ( int i = 0; i < Qinfo::numSimGroup(); ++i ) {
		const SimGroup* sg = Qinfo::simGroup( i );
		if ( sg->inQisReady() ) {
			/**
			 * Issue here. I would like to use a broadcast, but it doesn't
			 * look like that can be done with a non-blocking call. 
			 * May have to do a serial set of iSends, or grin and
			 * bear it with the blocking call.
			 * There is also the blocking Alltoall call. May be faster
			 * because it is low-level, but it then does not interleave
			 * computation with communication. But it is by far the 
			 * simplest.
			 */
			/*
			for ( int destNode = 0; destNode < numNodes_; ++destNode ) {
				pendingSends.push_back( sg->comm()->Isend(
					Qinfo::getInQ( i ), Qinfo::getInQsize( i ), 
					MPI::CHAR, destNode, sg->tag() ));
			}
			// OR
			sg->comm()->Bcast( 
				Qinfo::getInQ( i ), Qinfo::getInQsize( i ),
				MPI::CHAR, myNode_ );
			*/
			// OR
			sg->comm()->Alltoall( 
				Qinfo::getInQ( i ), sg->msgSize(), MPI::CHAR, 
				Qinfo::getMpiQ( i ), sg->msgSize(), MPI::CHAR );

			sg->releaseInQ(); // Now the threads can alter inQ.
		}
	}
}
#endif

/*
void Shell::startMpiThread()
{
	pthread_t mpiThread;
	pthread_attr_t attr;

	pthread_attr_init( &attr );
	pthread_attr_setdetachstate( &attr, PTHREAD_CREATE_JOINABLE );
	int ret = pthread_create( 
		&mpiThread, NULL, Shell::mpiThreadFunc, 
			reinterpret_cast< void* >( this )
	);
	if ( ret ) {
		cout << "Error: Shell::start: Unable to create mpi_threadn";
		exit( -1 );
	}

	// Clean up.
	void* status;
	int ret = pthread_join( mpiThread, &status );
	if ( ret ) {
		cout << "Error: Shell::start: Unable to join mpi_thread\n";
		exit( -1 );
	}
	pthread_attr_destroy( &attr );
}
*/

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

////////////////////////////////////////////////////////////////////////
// Functions for setting off clocked processes.
////////////////////////////////////////////////////////////////////////

void Shell::initThreadInfo( vector< ThreadInfo >& ti, 
	Element* clocke, Qinfo* q,
	pthread_mutex_t* sortMutex,
	double runtime )
{
	unsigned int j = 0;
	unsigned int numThreads = Qinfo::simGroup( 1 )->numThreads;
	if ( numNodes_ > 1 )
		numThreads++;

	/*
	cout << myNode_ << 
		": Shell::initThreadInfo: ti.size() = " << ti.size() << 
		", numCores = " << numCores_ << 
		", numSimGroup = " << Qinfo::numSimGroup() <<
		", numThreads[sg[1]] = " << Qinfo::simGroup( 1 )->numThreads <<
		", numThreads = " << numThreads <<
		endl;
		*/
	for ( unsigned int i = 1; i < Qinfo::numSimGroup(); ++i ) {
		for ( unsigned short k = 0; k < numThreads; ++k )
	//	for ( unsigned short k = 0; k < ti.size(); ++k )
		{
			ti[j].clocke = clocke;
			ti[j].qinfo = q;
			ti[j].runtime = runtime;
			ti[j].threadId = j;
			if ( i > 0 && numNodes_ > 1 && (k == numThreads - 1 ) ) {
				//cout << myNode_ << ": Shell::initThreadInfo: isMpiThread\n";
				ti[j].threadIndexInGroup = ~0; // isMpiThread
			} else {
				ti[j].threadIndexInGroup = 
					j - Qinfo::simGroup( i )->startThread + 1;
			}
			ti[j].nodeIndexInGroup = myNode_;
			ti[j].groupId = i;
			ti[j].outQid = Qinfo::simGroup(i)->startThread + k;
			ti[j].sortMutex = sortMutex;
			// cout << i << " " << k << ", j = " << j << ", thrIndexInGrp= " << ti[j].threadIndexInGroup << endl;
			j++;
		}
	}

	assert( j == ti.size() );
	assert( j == ( numCores_ + (numNodes_ > 1) ) );
}

void Shell::start( double runtime )
{
	Id clockId( 1 );
	Element* clocke = clockId();
	Qinfo q;
	if ( isSingleThreaded_ ) {
		// SetGet< double >::set( clocke, runTime );
		// clock->start( clocke, &q, runTime );
		Clock *clock = reinterpret_cast< Clock* >( clocke->dataHandler()->data( 0 ) );
		clock->start( clockId.eref(), &q, runtime );
		return;
	}

	unsigned int numThreads = numCores_;
#if USE_NODES
	if ( numNodes_ > 1 )
		++numThreads;
#endif

	vector< ThreadInfo > ti( numThreads );
	pthread_mutex_t sortMutex;
	pthread_mutex_init( &sortMutex, NULL );

	initThreadInfo( ti, clocke, &q, &sortMutex, runtime );

	setRunning( 1 );

	pthread_t* threads = new pthread_t[ numThreads ];
	pthread_attr_t attr;

	pthread_attr_init( &attr );
	pthread_attr_setdetachstate( &attr, PTHREAD_CREATE_JOINABLE );
	pthread_barrier_t barrier;
	if ( pthread_barrier_init( &barrier, NULL, numThreads ) ) {
		cout << "Error: Shell::start: Unable to init barrier\n";
		exit( -1 );
	}
	Clock* clock = reinterpret_cast< Clock* >( clocke->dataHandler()->data( 0 ) );
	clock->setBarrier( &barrier );
	barrier_ = &barrier;
	clock->setNumPendingThreads( 0 ); // Used for clock scheduling
	clock->setNumThreads( numThreads ); // Used for clock scheduling
	for ( unsigned int i = 0; i < numThreads; ++i ) {
		int ret = pthread_create( 
			&threads[i], NULL, Clock::threadStartFunc, 
			reinterpret_cast< void* >( &ti[i] )
		);
		if ( ret ) {
			cout << "Error: Shell::start: Unable to create threads\n";
			exit( -1 );
		}
	}
	// cout << "Shell::start: setting off with " << numNodes_ << " nodes on node " << myNode_ << "\n";

	// Clean up.
	for ( unsigned int i = 0; i < numThreads ; ++i ) {
		void* status;
		int ret = pthread_join( threads[ i ], &status );
		if ( ret ) {
			cout << "Error: Shell::start: Unable to join threads\n";
			exit( -1 );
		}
	}
		// cout << "Shell::start: Threads joined successfully\n";
		// cout << "Completed time " << runtime << " on " << numCores_ << " threads\n";
	
	static const Finfo* temp = Shell::initCinfo()->findFinfo( "ackStart" );
	static const SrcFinfo0* ackStart = 
		static_cast< const SrcFinfo0* >( temp );
	ackStart->send( Id().eref(), &p_, 0 );

	delete[] threads;
	pthread_attr_destroy( &attr );
	pthread_barrier_destroy( &barrier );
	pthread_mutex_destroy( &sortMutex );
}
