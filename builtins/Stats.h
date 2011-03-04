/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2011 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/
#ifndef _STATS_H
#define _STATS_H

class Stats
{
	public: 
		Stats();

		/**
 		 * Inserts an event into the pendingEvents queue for spikes.
 		 */
		void addSpike( DataId synIndex, const double time );
		
		////////////////////////////////////////////////////////////////
		// Field assignment stuff.
		////////////////////////////////////////////////////////////////
		
		double getMean() const;
		double getSdev() const;
		double getSum() const;
		unsigned int getNum() const;

		////////////////////////////////////////////////////////////////
		// Dest Func
		////////////////////////////////////////////////////////////////
		
		void trig( const Eref& e, const Qinfo* q );
		void process( const Eref& e, ProcPtr p );
		void reinit( const Eref& e, ProcPtr p );

		////////////////////////////////////////////////////////////////
		// Reduce func
		////////////////////////////////////////////////////////////////
		void digest( const Eref& er, const ReduceStats* arg );

		static const Cinfo* initCinfo();
	private:
		double mean_;
		double sdev_;
		double sum_;
		unsigned int num_;
};

#endif // _STATS_H
