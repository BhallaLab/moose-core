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
#ifdef USE_MPI
#include <mpi.h>
#endif
#include "header.h"
#include "ReduceFinfo.h"
#include "ReduceMsg.h"
#include "Shell.h"
#include "Dinfo.h"

#define USE_NODES 1

///////////////////////////////////////////////////////////////////
// First we have a few funcs which deal with acks coming in from
// different nodes to indicate completion of a function.
///////////////////////////////////////////////////////////////////
/**
 * Initialize acks. This call should be done before the 'send' goes out,
 * because with the wonders of threading we might get a response to the
 * 'send' before this call is executed.
 * This MUST be followed by a waitForAck call.
 */
void Shell::initAck()
{
	if ( isSingleThreaded() || !keepLooping() ) {
		numAcks_ = 0; 
	} else {
		pthread_mutex_lock( parserMutex() );
			// Note that we protect this in the mutex in the threaded mode.
			numAcks_ = 0;
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
	if ( isSingleThreaded() || !keepLooping() ) {
		while ( isAckPending() ) {
			Qinfo::clearQ( p_.threadIndexInGroup );
	// Tried this to get it to work in single-thread mode. Doesn't work.
	// Also causes problems by setting the size of Qinfo::reduceQ in
	// Qinfo::clearReduceQ
	//		Clock::checkProcState();
		}
	} else {
		while ( isAckPending() )
			pthread_cond_wait( parserBlockCond(), parserMutex() );
		isBlockedOnParser_ = 0;
		pthread_mutex_unlock( parserMutex() );
	}
}

/**
 * Test for receipt of acks from all nodes
 */ 
bool Shell::isAckPending() const
{
	// Put in timeout check here. At this point we would inspect the
	// acked vector to see which is last.
	return ( numAcks_ < numNodes_ );
}

/**
 * Generic handler for ack msgs from various nodes. Keeps track of
 * which nodes have responded.
 * The value is used optionally in things like getVec, to return # of
 * entries.
 */
void Shell::handleAck( unsigned int ackNode, unsigned int status )
{
	assert( ackNode < numNodes_ );
	acked_[ ackNode ] = status;
		// Here we could also check which node(s) are last, in order to do
		// some dynamic load balancing.
	++numAcks_;
	if ( status != OkStatus ) {
		cout << myNode_ << ": Shell::handleAck: Error: status = " <<
			status << " from node " << ackNode << endl;
	}
	if ( !isAckPending() && !isSingleThreaded() ) {
		pthread_cond_signal( parserBlockCond() );
	}
}

///////////////////////////////////////////////////////////////////

/**
 * Launches Parser. Blocking when the parser blocks.
 */
void Shell::launchParser()
{
	Id shellId;
	Shell* s = reinterpret_cast< Shell* >( shellId.eref().data() );
	bool quit = 0;
	
	cout << "moose : " << flush;
	while ( !quit ) {
		string temp;
		cin >> temp;
		if ( temp == "quit" || temp == "q" ) {
			s->doQuit();
			quit = 1;
		}
	}
	cout << "\nQuitting Moose\n" << flush;
}

// Function to assign hardware availability
void Shell::setHardware( 
	unsigned int numThreads, unsigned int numCores, unsigned int numNodes,
	unsigned int myNode )
{
	if ( numNodes > 1 )
		Qinfo::initMpiQs();
	numProcessThreads_ = numThreads;
	numCores_ = numCores;
	numNodes_ = numNodes;
	Qinfo::initQs( numThreads + 1, 1024 );
	myNode_ = myNode;
	p_.numNodesInGroup = numNodes_;
	p_.nodeIndexInGroup = myNode;
	p_.groupId = 0;
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
}

unsigned int Shell::numCores()
{
	return numCores_;
}

unsigned int Shell::numProcessThreads()
{
	return numProcessThreads_;
}

unsigned int Shell::numNodes()
{
	return numNodes_;
}

unsigned int Shell::myNode()
{
	return myNode_;
}

pthread_mutex_t* Shell::parserMutex()
{
	return parserMutex_;
}
pthread_cond_t* Shell::parserBlockCond()
{
	return parserBlockCond_;
}

bool Shell::inBlockingParserCall()
{
	return isBlockedOnParser_;
}

bool Shell::isSingleThreaded()
{
	return ( numProcessThreads_ == 0 );
}

bool Shell::keepLooping()
{
	return keepLooping_;
}


////////////////////////////////////////////////////////////////////////
// Functions for setting off clocked processes.
////////////////////////////////////////////////////////////////////////

void Shell::start( double runtime )
{
	/* Send msg to Clock
	*/
}

////////////////////////////////////////////////////////////////////////
// Functions using MPI
////////////////////////////////////////////////////////////////////////

void Shell::handleSync( const Eref& e, const Qinfo* q, Id elm, FuncId fid )
{

	assert( elm != Id() && elm() != 0 );
	/*
	FieldDataHandlerBase* fdh = dynamic_cast< FieldDataHandlerBase *>(
		elm()->dataHandler() );
		*/
	const ReduceFinfoBase* rfb = reduceArraySizeFinfo();

	if ( rfb )  {
		Msg * m = new ReduceMsg( Msg::setMsg, e, elm(), rfb );
		reduceMsg_ = m->mid();
		shelle_->addMsgAndFunc( m->mid(), fid, rfb->getBindIndex() );
		if ( myNode_ == 0 )
			rfb->send( Eref( shelle_, 0 ), ScriptThreadNum, 0 );
	}
	// We don't send an ack, instead the digest function does the update.
	// ack()->send( e, &p_, Shell::myNode(), OkStatus );
}
