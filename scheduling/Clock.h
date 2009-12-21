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

class Clock: public Data
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
		void setStage( DataId i, unsigned int  v );
		unsigned int  getStage( DataId i ) const;
		
		//////////////////////////////////////////////////////////
		//  Dest functions
		//////////////////////////////////////////////////////////
		void start( Eref e, const Qinfo* q, double runTime );
		void step( Eref e, const Qinfo* q, unsigned int nsteps );
		void stop( Eref e, const Qinfo* q );
		void reinit( Eref e, const Qinfo* q );

		///////////////////////////////////////////////////////////
		// Tick handlers
		///////////////////////////////////////////////////////////
		// Handles dt assignment from the child ticks.
		void setDt( Eref e, const Qinfo* q, double dt );

		void process( const ProcInfo* p, const Eref& e );

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

		/**
		 * Looks up the tick Element. Failure is not an option.
		 */
		Element* getTickE( Element* clocke );

//		void sortTicks();

		unsigned int getNumTicks() const;
		void setNumTicks( unsigned int num );

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
		int callback_;
		vector< TickPtr > tickPtr_;
		vector< Tick > ticks_;
};

class ThreadInfo
{
	public:
		Element* clocke;
		Qinfo* qinfo;
		double runtime;
		unsigned int threadId;
};

#endif // _CLOCK_H
