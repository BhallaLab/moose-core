/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _Tick_h
#define _Tick_h
class Tick
{
	public:
		Tick()
			: callback_( 0 ), dt_( 1.0 ), stage_( 0 ), nextTime_( 0.0 ),
				nextTickTime_( 0.0 ), next_( 0 )
		{
			;
		}

		virtual ~Tick( )
		{ ; }

		bool operator<( const Tick& other ) const;

		bool operator==( const Tick& other ) const;

		///////////////////////////////////////////////////////
		// Functions for handling field assignments.
		///////////////////////////////////////////////////////
		static void setDt( const Conn* c, double v );
		static double getDt( Eref e );
		static void setStage( const Conn* c, int v );
		static int getStage( Eref e );
		static int getOrdinal( Eref e );
		static double getNextTime( Eref e );

		static void setPath( const Conn* c, string v );
		static string getPath( Eref e );

		///////////////////////////////////////////////////////
		// Functions for handling messages
		///////////////////////////////////////////////////////

		static void receiveNextTime( const Conn* c, double v );
		static void incrementTick( const Conn* c, ProcInfo p, double v);
		void innerIncrementTick( Eref e, ProcInfo p, double v );

		/**
 		 * Resched is used to rebuild the scheduling. It does NOT mean that
 		 * the timings have to be updated: we may need to resched during a
 		 * run without missing a beat.
		 */
		static void resched( const Conn* c);

		/**
 		 * The innerResched function does two things: It sorts out the 
		 * ordering of the sequencing between ticks, and it may juggle
		 * around the ordering of calls to scheduled objects. The first
		 * task is handled by updateNextTickTime. The second task is
		 * tick class specific.
		 * For example, parTicks use this
		 * to decide which objects get scheduled for outgoingProcess and
		 * which remain on the local node. Yet more gory things may happen
		 * for multithreading. The base Tick class does not worry about
 		 * such details.
 		 */
		virtual void innerResched( const Conn* c);

		/**
 		 * updateNextTickTime cascades down the ticks to initialize their
 		 * nextTime_ field by querying the next one.
 		 * Invoked whenever there is a rescheduling.
 		 */
		void updateNextTickTime( Eref e );

		/**
		 * Reinit is used to set the simulation time back to zero for
		 * itself, and to trigger reinit in all targets, and to go on
		 * to the next tick
		 */
		static void reinit( const Conn* c, ProcInfo p );
		
		/**
		 * ReinitClock is used to reinit the state of the scheduling system.
		 * This does not send out reinit calls to objects connected to ticks.
		 */
		static void reinitClock( const Conn* c );

		static void handleNextTimeRequest( const Conn* c );

		static void start( const Conn* c, ProcInfo p, double maxTime );
		virtual void innerStart( Eref e, ProcInfo p, double maxTime );

		static void handleStop( const Conn* c, int callbackFlag );
		static void handleStopCallback( const Conn* c, int callbackFlag );

		static void handleCheckRunning( const Conn* c );
		static void handleRunningCallback( const Conn* c, bool isRunning );
		///////////////////////////////////////////////////////
		// Utility function
		///////////////////////////////////////////////////////
		int ordinal() const {
				return ordinal_;
		}
		///////////////////////////////////////////////////////
		// Virtual functions for handling scheduling. The
		// derived ParTick class puts in its own versions.
		///////////////////////////////////////////////////////
		virtual void innerProcessFunc( Eref e, ProcInfo info );
		virtual void innerReinitFunc( Eref e, ProcInfo info );

	private:
		bool running_;
		int callback_;
		double dt_;
		int stage_;
		double nextTime_;
		double nextTickTime_;
		bool next_; /// Flag to show if next_ tick is present
		bool terminate_;
		int ordinal_;
		string path_;/// \todo Perhaps we delete this field
		static int ordinalCounter_;
};

extern const Cinfo* initTickCinfo();

#endif // _Tick_h
