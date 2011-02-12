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
// Should be derived from projection, since we need to maintain a 
// text wildcard list of targets, and possibly manipulate it.
class Tick
{
	friend void testTicks();
	public:
		Tick();
		virtual ~Tick();

		bool operator<( const Tick& other ) const;
		bool operator==( const Tick& other ) const;

		///////////////////////////////////////////////////////
		// Functions for handling field assignments.
		///////////////////////////////////////////////////////
		void setDt( double v );
		double getDt() const;
		void setPath( string v );
		string getPath() const;

		///////////////////////////////////////////////////////
		// Functions for handling messages
		///////////////////////////////////////////////////////

		/**
		 * New version of 'advance'
 		 * This function is called to advance this one tick through one
		 * 'process' cycle. It is called in parallel on multiple threads,
		 * and the thread information is in the ProcInfo. It is the job of
		 * the target Elements to examine the ProcInfo and assign subsets
		 * of the object array to do the process operation.
		 */
		void advance( ProcInfo* p ) const;

		/**
		 * This assigns the index of this Tick in the array. This index
		 * is used to specify the Connection Slot to use for the outgoing
		 * clearQ and Process calls.
		 */
		void setIndex( unsigned int index );

		/**
		 * This assigns the parent Element of this Tick.
		 * It is used by the Tick to iterate through target messages
		 * when calling Process.
		 */
		void setElement( const Element* e );

		/**
		 * Reinit is used to set the simulation time back to zero for
		 * itself, and to trigger reinit in all targets, and to go on
		 * to the next tick
		void reinit( const Eref& e, ProcInfo* p ) const;
		 */

		/**
		 * Different version of reinit function. Designed for being called
		 * in multithread context, where ProcInfo carries info for current
		 * thread.
		 */
		void reinit( ProcInfo* p ) const;

		/**
		 * A dummy function for handling messages incoming from parent
		 * Clock
		 */
		void destroy( Eref e, const Qinfo* q );

		///////////////////////////////////////////////////////
		// Number of allocated ticks.
		static const unsigned int maxTicks;

		static const Cinfo* initCinfo();
	private:
		// bool running_;
		// int callback_;
		double dt_;

		/**
		 * This is the index of this Tick in the Tick array. It is needed 
		 * to rapidly look up the correct Connection Slot to call for 
		 * Process and clearQ.
		 */
		unsigned int index_; 
// 		string path_;/// \todo Perhaps we delete this field

		/**
		 * The object needs to keep track of its own Element in order
		 * to traverse the process message list.
		 */
		const Element* ticke_;
};

#endif // _Tick_h
