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
		enum ProcState { 
			NoChange, 
			TurnOnReinit,
			TurnOffReinit,
			ReinitThenStart, 
			StartOnly, 
			StopOnly, 
			StopThenReinit,
			QuitProcessLoop
		};
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
		 * Halts running simulation gracefully, letting it restart 
		 * wherever it left off from.
		 */
		void stop();

		/**
		 * Quit the main event loop gracefully. It will wait till all
		 * threads and nodes are at barrier3 before going.
		 */
		void handleQuit();

		/**
		 * This utility function creates a tick on the assigned tickNum,
		 * Assigns dt.
		 * Must only be called at a thread-safe time.
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
		/// dest function for message to run simulation for specified time
		void handleStart( double runtime );

		/// dest function for message to run simulation for specified steps
		void handleStep( unsigned int steps );

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

		/**
		 * Flag: True when the Process loop is still going around
		 */
		bool keepLooping() const;
		/**
		 * Assign state for Process loop
		 */
		void setLoopingState( bool val );

		/**
		 * Flag: True when the simulation is still running.
		 */
		bool isRunning() const;

		/**
		 * Diagnostic: reports the numbe rof times around different phases
		 * of the Process loop
		 */
		void printCounts() const;

		/**
		 * Static function, used to flip flags to start or end simulation. 
		 * It is used as the within-barrier function of barrier 3.
		 * This has to be in the barrier as we are altering Clock fields
		 * doingReinit_ and isRunning_,
		 * which the 'process' operation depends on.
		 */
		static void checkProcState();

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
		TickMgr* tp0_;
		 */

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
		 * Global flag to tell Clock how to change process state next cycle
		 */
		static ProcState procState_;

};

#endif // _CLOCK_H
