/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "Tracker.h"

Tracker::Tracker( int numNodes, int numThreads, Rule rule )
	: 
		numHops_( 0 ),
		numNodes_( numNodes ),
		numThreads_( numThreads ),
		trajectoryRule_( rule )
{
	for ( unsigned int i = 0; i < HistorySize; ++i ) {
		recentNodes_[i] = recentThreads_[i] = 0;
	}

	for ( unsigned int i = 0; i < maxNodes; ++i ) {
		for ( unsigned int j = 0; j < maxThreads; ++j ) {
			touchdowns_[ i ][ j ] = 0;
		}
	}
}

// Returns 0 if it has landed in an incorrect place.
bool Tracker::updateHistory( int node, int thread )
{
	if ( numHops_ == 0 ) {
		firstNode_ = node;
		firstThread_ = thread;
	} else {
		int expectedNode = 0;
		int expectedThread = 0;

		--numHops_; // go back to check history
		nextHop( expectedNode, expectedThread );
		++numHops_; // return to present.
		if ( expectedNode != node || expectedThread != thread )
			return 0;
	}
	recentNodes_[ numHops_ % HistorySize ] = node;
	recentThreads_[ numHops_ % HistorySize ] = thread;
	++numHops_;
	return 1;
}

// Currently always returns 1
bool Tracker::nextHop( int& nextNode, int& nextThread )
{
	int lastNode = recentNodes_[ numHops_ % HistorySize ];
	int lastThread = recentThreads_[ numHops_ % HistorySize ];
	switch ( trajectoryRule_ ) {
		case raster0: 
			nextThread = lastThread + 1;
			if ( nextThread == numThreads_ ) {
				nextThread = 0;
				nextNode = lastNode + 1;
				if ( nextNode == numNodes_ )
					nextNode = 0;
			}
			break;
		case raster90: 
			nextNode = lastNode + 1;
			if ( nextNode == numNodes_ ) {
				nextNode = 0;
				nextThread = lastThread + 1;
				if ( nextThread == numThreads_ )
					nextThread = 0;
			}
			break;
		case raster180: 
			nextThread = lastThread - 1;
			if ( nextThread < 0 ) {
				nextThread = numThreads_ - 1;
				nextNode = lastNode - 1;
				if ( nextNode < 0 )
					nextNode = numNodes_ - 1;
			}
			break;
		case raster270: 
			nextNode = lastNode - 1;
			if ( nextNode < 0 ) {
				nextNode = numNodes_ - 1;
				nextThread = lastThread - 1;
				if ( nextThread < 0 )
					nextThread = numThreads_ - 1;
			}
			break;
	}
	return 1;
}
