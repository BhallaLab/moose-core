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
		unsigned int getNumPendingThreads() const;
		void setNumPendingThreads( unsigned int num );
		unsigned int getNumThreads() const;
		void setNumThreads( unsigned int num );
		
		//////////////////////////////////////////////////////////
		//  Dest functions
		//////////////////////////////////////////////////////////
		/**
		 * starts up a run to go for runTime without threading.
		 */
		void start( const Eref& e, const Qinfo* q, double runTime );

		/**
		 * tStart starts up a run using threading. Is called independently
		 * on each worker thread. threadId starts from 0 and goes up to
		 * # of worker threads. threadId 0 has a special meaning as it 
		 * manages increments of current time.
		 */
		// void tStart(  const Eref& e, const Qinfo* q, double runTime, unsigned int threadId );
		void tStart(  const Eref& e, const ThreadInfo* ti );
		void sortTickPtrs( pthread_mutex_t* sortMutex );
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

		unsigned int getNumTicks() const;
		void setNumTicks( unsigned int num );
		void setBarrier( void* barrier1, void* barrier2 );

		static void* threadStartFunc( void* threadInfo );
		static const Cinfo* initCinfo();
	private:
		double runTime_;
		double currentTime_;
		double nextTime_;
		unsigned int nSteps_;
		unsigned int currentStep_;
		double dt_; /// The minimum dt among all ticks.
		bool isRunning_;
		ProcInfo info_;
		unsigned int numPendingThreads_;
		unsigned int numThreads_;
		int callback_;
		vector< TickPtr > tickPtr_;
		vector< Tick > ticks_;
		TickPtr* tp0_;
};

#endif // _CLOCK_H
