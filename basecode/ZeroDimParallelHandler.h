/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2012 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _ZERO_DIM_PARALLEL_HANDLER_H
#define _ZERO_DIM_PARALLEL_HANDLER_H

/**
 * This class handles objects which remain singular in multithread/MPI
 * mode, but which nevertheless need to receive calls on many threads.
 * Typical examples are solvers that internally manage the partitioning
 * among threads, and need to be updated on many threads during the 
 * process call.
 * Specifically, this Handler passes the job of deciding what to do on
 * each thread down to the object. The object gets called on all threads
 * and must make thread-safe operations internally.
 */
class ZeroDimParallelHandler: public ZeroDimHandler
{
	public:
		/// This is the generic constructor
		ZeroDimParallelHandler( const DinfoBase* dinfo, 
			const vector< DimInfo >& dims, unsigned short pathDepth,
			bool isGlobal );

		/// Special constructor used in Cinfo::makeCinfoElements
		ZeroDimParallelHandler( const DinfoBase* dinfo, char* data );

		/// This is the copy constructor
		ZeroDimParallelHandler( const ZeroDimParallelHandler* other );

		~ZeroDimParallelHandler();

		////////////////////////////////////////////////////////////
		// Information functions: all inherited.
		////////////////////////////////////////////////////////////
		////////////////////////////////////////////////////////////////
		// load balancing functions
		////////////////////////////////////////////////////////////////
		bool execThread( ThreadId thread, DataId di ) const;
		////////////////////////////////////////////////////////////////
		// Process function
		////////////////////////////////////////////////////////////////
		/**
		 * calls process on data, using threading info from the ProcInfo
		 */
		void process( const ProcInfo* p, Element* e, FuncId fid ) const;

		/**
		 * Calls OpFunc f on all data entries, using threading info from 
		 * the Qinfo and the specified argument(s)
		 */
		void forall( const OpFunc* f, Element* e, const Qinfo* q,
			const double* arg, unsigned int argSize, unsigned int numArgs )
			const;

		////////////////////////////////////////////////////////////////
		// Data Reallocation functions
		////////////////////////////////////////////////////////////////
		/**
		 * Make a single identity copy, doing appropriate node 
		 * partitioning if toGlobal is false.
		 */
		DataHandler* copy( unsigned short newParentDepth,
			unsigned short copyRootDepth,
			bool toGlobal, unsigned int n ) const;

		DataHandler* copyUsingNewDinfo( const DinfoBase* dinfo) const;

	private:
};

#endif // _ZERO_DIM_PARALLEL_HANDLER_H
