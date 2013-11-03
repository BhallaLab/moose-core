/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _CLOCK_H
#define _CLOCK_H

/**
 * Clock now uses integral scheduling. The Clock has an array of child
 * Ticks, each of which controls the process and reinit calls of its 
 * targets. The Clock has a minimum dt.
 * All ticks operate with (positive) integral multiples of this, 
 * from 1 to anything. If multiple Ticks are due to go off in a given
 * cycle, the order is from lowest to highest. Within a Tick the order
 * of execution of target objects is undefined.
 *
 * The Reinit call goes through all Ticks in order.
 */

class Clock
{
	friend void testClock();
	public:
		Clock();

		//////////////////////////////////////////////////////////
		//  Field assignment functions
		//////////////////////////////////////////////////////////
		void setDt( double v );
		double getDt() const;
		double getRunTime() const;
		double getCurrentTime() const;
		unsigned int getNsteps( ) const;
		unsigned int getCurrentStep() const;

		void setTickStep( unsigned int i, unsigned int v );
		unsigned int getTickStep( unsigned int i ) const;
		void setTickDt( unsigned int i, double v );
		double getTickDt( unsigned int i ) const;

		vector< double > getDts() const;
		
		//////////////////////////////////////////////////////////
		//  Dest functions
		//////////////////////////////////////////////////////////
		/**
		 * Halts running simulation gracefully, letting it restart 
		 * wherever it left off from.
		 */
		void stop();

		/// dest function for message to run simulation for specified time
		void handleStart( const Eref& e, double runtime );

		/// dest function for message to run simulation for specified steps
		void handleStep( const Eref& e, unsigned int steps );

		/// dest function for message to trigger reinit.
		void handleReinit( const Eref& e );

		///////////////////////////////////////////////////////////////
		// Stuff for new scheduling.
		///////////////////////////////////////////////////////////////
		// Goes through the process loop.
		void process();


		///////////////////////////////////////////////////////////////
		unsigned int getNumTicks() const;

		/**
		 * Flag: True when the simulation is still running.
		 */
		bool isRunning() const;
		/**
		 * Flag: True when the simulation is busy with reinit
		 */
		bool isDoingReinit() const;

		/**
		 * Utility function to tell us about the scheduling
		 */
		static void reportClock();
		void innerReportClock() const;

		// static void* threadStartFunc( void* threadInfo );
		static const Cinfo* initCinfo();

		/// Utility func to range-check when Ticks are being changed.
		bool checkTickNum( const string& funcName, unsigned int i ) const;

	private:
		void buildTicks( const Eref& e );
		double runTime_;
		double currentTime_;
		unsigned int nSteps_;
		unsigned int currentStep_;
		double dt_; /// The minimum dt. All ticks are a multiple of this.

		/**
		 * True while a process job is running
		 */
		bool isRunning_;

		/**
		 * True while the system is doing a reinit
		 */
		bool doingReinit_;

		/**
		 * Maintains Process info
		 */
		ProcInfo info_;

		/**
		 * Ticks are sub-elements and send messages to sets of target
		 * Elements that handle calculations in which update order does
		 * not matter. 
		 */
		vector< unsigned int > ticks_;

		/**
		 * Array of active ticks. Drops out ticks lacking targets or with 
		 * a zero step. Sorted in increasing order of tick index.
		 */
		vector< unsigned int > activeTicks_;

		/**
		 * number of Ticks.
		 */
		static const unsigned int numTicks;
};

#endif // _CLOCK_H
