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
			: dt_( 1.0 ), stage_( 0 ), nextTime_( 0.0 ),
				nextTickTime_( 0.0 ), next_( 0 )
		{
			;
		}

		virtual ~Tick( )
		{ ; }

		bool operator<( const Tick& other ) const {
			if ( dt_ < other.dt_ ) return 1;
			if ( dt_ == other.dt_ && stage_ < other.stage_ )
				return 1;
			return 0;
		}

		bool operator==( const Tick& other ) const {
			return ( dt_ == other.dt_ && 
				stage_ == other.stage_ );
		}

		///////////////////////////////////////////////////////
		// Functions for handling field assignments.
		///////////////////////////////////////////////////////
		static void setDt( const Conn& c, double v );
		static double getDt( const Element* e );
		static void setStage( const Conn& c, int v );
		static int getStage( const Element* e );
		static int getOrdinal( const Element* e );
		static double getNextTime( const Element* e );

		static void setPath( const Conn& c, string v );
		static string getPath( const Element* e );

		///////////////////////////////////////////////////////
		// Functions for handling messages
		///////////////////////////////////////////////////////

		static void receiveNextTime( const Conn& c, double v );
		static void incrementTick( const Conn& c, ProcInfo p, double v);
		void innerIncrementTick( Element* e, ProcInfo p, double v );
		static void resched( const Conn& c);
		void updateNextTickTime( Element* e );
		static void reinit( const Conn& c, ProcInfo p );

		static void handleNextTimeRequest( const Conn& c );

		static void start( const Conn& c, ProcInfo p, double maxTime );
		void innerStart( Element* e, ProcInfo p, double maxTime );
		static void schedNewObject( const Conn& c, unsigned int id,
						string s );
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
		virtual void innerProcessFunc( Element* e, ProcInfo info );
		virtual void innerReinitFunc( Element* e, ProcInfo info );

	private:
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
