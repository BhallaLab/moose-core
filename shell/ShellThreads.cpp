/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

/**
 * This file contains the MPI-handling functions in Shell.
 */

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
	numAcks_ = 0; 
}

/**
 * test for completion of request. This MUST be preceded by an initAck
 * call.
 */
void Shell::waitForAck()
{
	while ( isAckPending() ) {
			Qinfo::clearQ( p_.threadIndexInGroup );
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
	assert( numThreads == 1 );
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

bool Shell::isParserIdle()
{
	return Shell::isParserIdle_;
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
	const ReduceFinfoBase* rfb = reduceArraySizeFinfo();

	if ( rfb )  {
		Msg * m = new ReduceMsg( Msg::setMsg, e, elm(), rfb );
		reduceMsg_ = m->mid();
		shelle_->addMsgAndFunc( m->mid(), fid, rfb->getBindIndex() );
		if ( myNode_ == 0 )
			rfb->send( Eref( shelle_, 0 ), ScriptThreadNum, 0 );
	}
}
