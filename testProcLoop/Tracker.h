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
		Tracker();

		Tracker( int numNodes, int numThreads, Rule rule );

		/**
		 * Assigns next thread/node to go to, and pushes it into its history.
		 */
		void setNextHop();

		/**
		 * Figure out next node and next thread.
		 */
		bool nextHop( int& nextNode, int& nextThread ) const;

		/**
		 * Return True if this is the end of the Track.
		 */
		bool stop() const;

		/**
		 * Assign stop field for Tracker
		 */
		void setStop( bool val );

		/** 
		 * Return current node it should be on
		 */
		int node() const;

		/** 
		 * Return current thread it should be on
		 */
		int thread() const;

		/**
		 * Prints out status of tracker
		 */
		void print() const;

	private:
		// Node:thread of last 10 visits
		int recentNodes_[ HistorySize ];
		int recentThreads_[ HistorySize ];

		unsigned int numHops_; // How many places has it visited

		int firstNode_;
		int firstThread_;

		// Sum of counts of of visits on each Node:thread combination
		// int touchdowns_[ maxNodes ][ maxThreads ];
		int numNodes_;
		int numThreads_;

		Rule trajectoryRule_;
		bool stop_;
};
