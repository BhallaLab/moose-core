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
			: Tick(), pendingCount_( 0 )
		{
			;
		}

		virtual ~ParTick( )
		{ ; }

		///////////////////////////////////////////////////////
		// Functions for DestMessages
		///////////////////////////////////////////////////////
		static void pollFunc( const Conn& c, unsigned int node );

		///////////////////////////////////////////////////////
		// Virtual functions for handling scheduling of PostMaster,
		// locally connected, and remote connected objects.
		///////////////////////////////////////////////////////
		void innerProcessFunc( Element* e, ProcInfo info );
		void innerReinitFunc( Element* e, ProcInfo info );

		///////////////////////////////////////////////////////
		// Utility function to set up the pending list.
		///////////////////////////////////////////////////////
		void initPending( Element* e );
		void innerPollFunc( unsigned int node );
		bool pendingData() const;

	private:
		vector< bool > pendingNodes_;
		unsigned int pendingCount_;
};

#endif // _ParTick_h
