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

class TickPtr {
	public:
		TickPtr()
			: ptr_( 0 )
		{;}
		
		TickPtr( Tick* ptr )
			: ptr_( ptr )
		{;}

		bool operator<( const TickPtr other ) const {
			if ( ptr_ && other.ptr_ )
				return ( *ptr_ < *( other.ptr_ ) );
			return 0;
		}

		const Tick* tick() const {
			return ptr_;
		}

		/**
		 * Advance the simulation till the specified end time, without
		 * worrying about other dts.
		 */
		void advance( Eref e, ProcInfo* p, double endTime ) {
			while ( p->currTime < endTime ) {
				const TickPtr* t = this;
				while ( t ) {
					t->ptr_->increment( e, p );
					t = t->next_;
				}
				p->currTime += p->dt;
			}
		}

		void reinit( Eref e ) const
		{
			const TickPtr* t = this;
			while ( t ) {
				t->ptr_->reinit( e );
				t = t->next_;
			}
		}

		
	private:
		Tick* ptr_;
		const TickPtr* next_;
};


class Clock: public Data
{
	public:
		Clock();

		void process( const ProcInfo* p, const Eref& e );

		//////////////////////////////////////////////////////////
		//  Field assignment functions
		//////////////////////////////////////////////////////////
		void setRunTime( double v );
		double getRunTime() const;
		double getCurrentTime() const;
		void setNsteps( unsigned int v );
		unsigned int getNsteps( ) const;
		unsigned int getCurrentStep() const;
		
		//////////////////////////////////////////////////////////
		//  Dest functions
		//////////////////////////////////////////////////////////
		void start( Eref e, const Qinfo* q, double runTime );
		void step( Eref e, const Qinfo* q, unsigned int nsteps );
		void stop( Eref e, const Qinfo* q );
		void reinit( Eref e, const Qinfo* q );

		// Handles dt assignment from the child ticks.
		void setDt( Eref e, const Qinfo* q, double dt );
		void sortTicks();

		static const Cinfo* initCinfo();
	private:
		double runTime_;
		double currentTime_;
		double nextTime_;
		int nSteps_;
		int currentStep_;
		double dt_;
		bool isRunning_;
		ProcInfo info_;
		int callback_;
		vector< TickPtr > tickPtr_;
		vector< Tick > ticks_;
};

#endif // _CLOCK_H
