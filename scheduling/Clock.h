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

class Clock
{
	friend void setupTicks();
	public:
		Clock();

		//////////////////////////////////////////////////////////
		//  Field assignment functions
		//////////////////////////////////////////////////////////
		void setRunTime( double v );
		double getRunTime() const;
		double getCurrentTime() const;
		void setNsteps( unsigned int v );
		unsigned int getNsteps( ) const;
		unsigned int getCurrentStep() const;

		void setTickDt( DataId i, double v );
		double getTickDt( DataId i ) const;
		
		//////////////////////////////////////////////////////////
		//  Dest functions
		//////////////////////////////////////////////////////////
		/**
		 * starts up a run to go for runTime without threading.
		 */
		void start( const Eref& e, const Qinfo* q, double runTime );
		void step( const Eref& e, const Qinfo* q, unsigned int nsteps );
		void stop( const Eref& e, const Qinfo* q );
		void terminate( const Eref& e, const Qinfo* q );
		void reinit( const Eref& e, const Qinfo* q );

		/**
		 * This utility function creates a tick on the assigned tickNum,
		 * Assigns dt.
		 */
		void setupTick( unsigned int tickNum, double dt );

		///////////////////////////////////////////////////////////
		// Tick handlers
		///////////////////////////////////////////////////////////
		// Handles dt assignment from the child ticks.
		void setDt( const Eref& e, const Qinfo* q, double dt );

		/**
		 * Pushes the new Tick onto the TickPtr stack.
		 */
		void addTick( Tick* t );

		/**
		 * Scans through all Ticks and puts them in order onto the tickPtr_
		 */
		void rebuild();

		/**
		 * Looks up the specified clock tick. Returns 0 on failure.
		 */
		Tick* getTick( unsigned int i );

		///////////////////////////////////////////////////////////////
		// Stuff for new scheduling.
		///////////////////////////////////////////////////////////////

		/**
		 * Advance system state by one clock tick. This may be a subset of
		 * one timestep, as there may be multiple clock ticks within one dt.
		 * This is meant to be run in parallel on multiple threads. The
		 * ProcInfo carries info about thread. 
		 */
		void advancePhase1( ProcInfo* p );
		void advancePhase2( ProcInfo* p );
		/// dest function for message to start simulation.
		void handleStart( double runtime );

		/**
		 * Reinit is used to reinit the state of the scheduling system.
		 * This version is meant to be done through the multithread 
		 * scheduling loop. It has to do this to initialize ProcInfo 
		 * properly.
		 */
		void reinitPhase1( ProcInfo* p );
		void reinitPhase2( ProcInfo* p );
		/// dest function for message to trigger reinit.
		void handleReinit();


		/**
		 * The process functions are the interface presented by the Clock
		 * to the multithread process loop.
		 */
		void processPhase1( ProcInfo* p );
		void processPhase2( ProcInfo* p );

		///////////////////////////////////////////////////////////////
		unsigned int getNumTicks() const;
		void setNumTicks( unsigned int num );
		void setBarrier( void* barrier1, void* barrier2 );

		bool keepLooping() const;
		void setLoopingState( bool val );

		void printCounts() const;

		/**
		 * Static function, used to flip flags to start or end simulation. 
		 * It is used as the within-barrier function of barrier 3.
		 * This has to be in the barrier as we are altering a Clock field
		 * which the 'process' flag depends on.
		 */
		static void checkStartOrStop();

		// static void* threadStartFunc( void* threadInfo );
		static const Cinfo* initCinfo();
	private:
		double runTime_;
		double currentTime_;
		double nextTime_;
		double endTime_;
		unsigned int nSteps_;
		unsigned int currentStep_;
		double dt_; /// The minimum dt among all ticks.

		/**
		 * True while a process job is running
		 */
		bool isRunning_;

		/**
		 * True while the system is doing a reinit
		 */
		bool doingReinit_;
		ProcInfo info_;
		unsigned int numPendingThreads_;
		unsigned int numThreads_;
		int callback_;

		/**
		 * True while main event loop continues
		 */
		bool keepLooping_;

		/**
		 * TickPtr contains pointers to tickMgr and is used to sort.
		 */
		vector< TickPtr > tickPtr_;

		/**
		 * TickMgr groups together Ticks with the same dt. Within this
		 * group, the Ticks are ordered by their index in the main ticks_
		 * vector. However, this order doesn't change so it does not
		 * need to be resorted every step increment.
		 */
		vector< TickMgr > tickMgr_;

		/**
		 * Ticks are sub-elements and send messages to sets of target
		 * Elements that handle calculations in which update order does
		 * not matter. 
		 */
		vector< Tick > ticks_;

		/**
		 * This points to the current TickMgr first in line for execution
		 */
		TickMgr* tp0_;

		/**
		 * Counters to keep track of number of passes through each process
		 * phase. Useful for debugging, and load balancing in due course.
		 */
		unsigned int countNull1_;
		unsigned int countNull2_;
		unsigned int countReinit1_;
		unsigned int countReinit2_;
		unsigned int countAdvance1_;
		unsigned int countAdvance2_;

		/**
		 * Global flag to tell Clock that it has to stop running. 
		 */
		static bool flipRunning_;
};

#endif // _CLOCK_H
