/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#define HistorySize 10
#define maxNodes 256
#define maxThreads 16

/**
 * The Tracker class is meant to be passed around as a message. It 
 * keeps track of where it has been and where it is to go next.
 * Current trajectories are defined by the 'Rule' which is a raster
 * scan in the specified angle. Raster0 goes along the +x coordinate,
 * then returns to (0, y+1) and continues till (numNodes, numThreads).
 * Then it loops back again. Other rasters are similar, with different
 * angles.
 */

enum Rule { raster0, raster90, raster180, raster270 };
class Tracker
{
	public:
		Tracker( int numNodes, int numThreads, Rule rule );

		/**
		 * Adds latest hop to history. It knows what to expect, so it
		 * returns 0 if it has ended up somewhere unexpected.
		 * On the first hop it is OK with any values to come in.
		 */
		bool updateHistory( int node, int thread );

		/**
		 * Using trajectory rule, decides where to go next. Returns 0 if
		 * it is to terminate.
		 */
		bool nextHop( int& nextNode, int& nextThread );
	private:
		// Node:thread of last 10 visits
		int recentNodes_[ HistorySize ];
		int recentThreads_[ HistorySize ];

		unsigned int numHops_; // How many places has it visited

		int firstNode_;
		int firstThread_;

		// Sum of counts of of visits on each Node:thread combination
		int touchdowns_[ maxNodes ][ maxThreads ];
		int numNodes_;
		int numThreads_;

		Rule trajectoryRule_;
};
