/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _ParTick_h
#define _ParTick_h
class ParTick: public Tick
{
	public:
		ParTick()
			: Tick(), pendingCount_( 0 ), barrier_( 0 ), doSync_( 0 )
		{
			;
		}

		virtual ~ParTick( )
		{ ; }

		///////////////////////////////////////////////////////
		// Functions for Fields
		///////////////////////////////////////////////////////
		static void setBarrier( const Conn* c, int v );
		static int getBarrier( Eref e );
		static void setSync( const Conn* c, bool v );
		static bool getSync( Eref e );

		///////////////////////////////////////////////////////
		// Functions for DestMessages
		///////////////////////////////////////////////////////
		static void pollFunc( const Conn* c, unsigned int node );

		///////////////////////////////////////////////////////
		// Virtual functions for handling scheduling of PostMaster,
		// locally connected, and remote connected objects.
		///////////////////////////////////////////////////////
		void innerProcessFunc( Eref e, ProcInfo info );
		void innerReinitFunc( Eref e, ProcInfo info );
		void innerResched( const Conn* c);
		void innerStart( Eref e, ProcInfo p, double maxTime );

		///////////////////////////////////////////////////////
		// Utility function to set up the pending list.
		///////////////////////////////////////////////////////
		void initPending( Eref e );
		void innerPollFunc( unsigned int node );
		bool pendingData() const;

		///////////////////////////////////////////////////////
		// Utility function for debugging.
		///////////////////////////////////////////////////////
		void printPos( const string& s );

	private:
		vector< bool > pendingNodes_; // Entries are true if node is pending
		unsigned int pendingCount_; // How many nodes remain to finish poll
		unsigned int numOutgoing_; // How many outgoing messages?
		bool barrier_; // True if this Tick should end with a barrier.

		/**
		 * True if this Tick should enforce synchronization of all nodes
		 * connected to this one, via the dest postmasters. 
		 * This should be true for Ticks mediating simulations.
		 * This should be false for Ticks handling setup.
		 */
		bool doSync_; 
};

#endif // _ParTick_h
