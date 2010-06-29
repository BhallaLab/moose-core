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

class TickPtr {
	friend void setupTicks();
	public:
		TickPtr();
		TickPtr( Tick* ptr );

		bool operator<( const TickPtr other ) const;

		/**
		 * Add a tick to the specified TickPtr, provided dt is OK.
		 * The new tick is positioning according to stage.
		 * This is inefficient, but we don't expect to have many ticks,
		 * typically under 10.
		 * Returns true if the dt matches and the add was successful.
		 */
		bool addTick( const Tick* t );

		// Won't bother with a clear function, just rebuild the whole lot.

		/**
		 * Advance the simulation till the specified end time, without
		 * worrying about other dts.
		 */
		void advance( Element* e, ProcInfo* p, double endTime );

		double getNextTime() const;
		double getDt() const;

		/**
		 * Set to time = 0. So, nextTime_ = dt.
		 */
		void reinit( const Eref& e, ProcInfo* p );
		
	private:
		// Tick* ptr_;
		// const TickPtr* next_;
		double dt_;
		double nextTime_; // Upcoming time
		vector< const Tick* > ticks_;	// Pointer to each Tick.
		static double EPSILON;
};

#endif // _TICKPTR_H
