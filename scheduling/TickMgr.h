/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _TICKPTR_H
#define _TICKPTR_H

class TickMgr {
	friend void setupTicks();
	public:
		TickMgr();
		TickMgr( Tick* ptr );

		/**
		 * Add a tick to the specified TickMgr, provided dt is OK.
		 * The new tick is positioning according to stage.
		 * This is inefficient, but we don't expect to have many ticks,
		 * typically under 10.
		 * Returns true if the dt matches and the add was successful.
		 */
		bool addTick( const Tick* t );

		///////////////////////////////////////////////////////////////
		// Stuff for new scheduling.
		///////////////////////////////////////////////////////////////

		/**
		 * New version of 'advance', used in the new multithread scheduling.
		 * Here we move ahead by one tick only. This may be a subset of
		 * one timestep if there are multiple ticks.
		 */
		void advancePhase1( ProcInfo* p ) const;
		void advancePhase2( ProcInfo* p );

		/**
		 * New version of 'reinit'. This happens in three phases.
		 * Zero: we set the nextTime to the dt.
		 * One: we send the reinit call out to all scheduled objects,
		 * Two: Advance to the next Tick, and when those are done, to 
		 * the next TickMgr.
		 */
		void reinitPhase0();
		void reinitPhase1( ProcInfo* p ) const;
		bool reinitPhase2( ProcInfo* p );


		///////////////////////////////////////////////////////////////
		double getNextTime() const;
		double getDt() const;

		/**
		 * Set to time = 0. So, nextTime_ = dt.
		void reinit( const Eref& e, ProcInfo* p );
		 */

		/**
		 * True if tickerator_ has not been set to something sensible
		 */
		bool isInited() const;

		/**
		 * Utility function to inspect tick contents
		 */
		const vector< const Tick* >& ticks() const;
		
	private:
		// Tick* ptr_;
		// const TickMgr* next_;
		double dt_;
		double nextTime_; // Upcoming time
		vector< const Tick* > ticks_;	// Pointer to each Tick.
		unsigned int tickerator_;
		// vector< const Tick* >::iterator tickerator_;
		static double EPSILON;
};

#endif // _TICKPTR_H
