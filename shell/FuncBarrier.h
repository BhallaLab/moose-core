/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _FUNC_BARRIER
#define _FUNC_BARRIER
/**
 * This class sets up a pthreads barrier with the capability to do
 * an atomic user-defined operation at the tail end of the barrier.
 * This is needed in the MOOSE Barrier1, where the inQ and the outQ
 * must swap within the barrier without being exposed to any of the
 * assorted threads.
 * The FuncBarrier has to be defined out of the scope of the threads
 * it is controlling. Likewise destroyed after they have joined.
 */

 class FuncBarrier 
 {
 	public:
		FuncBarrier( unsigned int numThreads, 
			void (*op)() = &FuncBarrier::dummyOp );
		~FuncBarrier();
		void wait();
		static void dummyOp();
	private:
		unsigned int numThreads_; /// Number of threads handled by barrier
		unsigned int threadsRemaining_; /// Number of threads still to clear
		unsigned int numLoops_;	/// Number of times barrier has been crossed

		pthread_mutex_t	mutex_;	/// Mutex used within barrier
		pthread_cond_t cond_;	/// Conditional used by barrier

		void (*op_)();			// Function to be executed within barrier.
 };

 #endif // _FUNC_BARRIER
