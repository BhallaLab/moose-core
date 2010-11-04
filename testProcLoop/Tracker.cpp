/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include <iostream>
#include "Tracker.h"

using namespace std;

Tracker::Tracker( int numNodes, int numThreads, Rule rule )
	: 
		numHops_( 0 ),
		numNodes_( numNodes ),
		numThreads_( numThreads ),
		trajectoryRule_( rule ),
		stop_( 0 )
{
	for ( unsigned int i = 0; i < HistorySize; ++i ) {
		recentNodes_[i] = recentThreads_[i] = 0;
	}

	/*
	for ( unsigned int i = 0; i < maxNodes; ++i ) {
		for ( unsigned int j = 0; j < maxThreads; ++j ) {
			touchdowns_[ i ][ j ] = 0;
		}
	}
	*/
}

Tracker::Tracker()
	:
		numHops_( 0 ),
		numNodes_( 0 ),
		numThreads_( 0 ),
		trajectoryRule_( raster0 ),
		stop_( 1 )
{;}

void Tracker::setNextHop()
{
	int nextNode;
	int nextThread;
	// ++touchdowns_[ node() ][ thread() ];
	nextHop( nextNode, nextThread );
	++numHops_;
	recentNodes_[ numHops_ % HistorySize ] = nextNode;
	recentThreads_[ numHops_ % HistorySize ] = nextThread;
}

// Currently always returns 1
bool Tracker::nextHop( int& nextNode, int& nextThread ) const
{
	int lastNode = node();
	int lastThread = thread();
	nextNode = lastNode;
	nextThread = lastThread;
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

bool Tracker::stop() const
{
	return stop_;
}

void Tracker::setStop( bool val ) 
{
	stop_ = val;
}

int Tracker::node() const
{
	return recentNodes_[ numHops_ % HistorySize ];
}

int Tracker::thread() const
{
	return recentThreads_[ numHops_ % HistorySize ];
}

void Tracker::print() const
{
	unsigned int lastIndex = 0;
	if ( numHops_ > 0 )
		lastIndex = (numHops_ - 1) % HistorySize;
	cout << "Tracker rule " << trajectoryRule_ << 
		" on (" << recentNodes_[ lastIndex ] << ":" <<
		recentThreads_[ lastIndex ] << "), hop=" << numHops_ << endl;
}

