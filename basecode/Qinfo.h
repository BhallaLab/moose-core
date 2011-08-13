/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef QINFO_H
#define QINFO_H

/// Forward declarations
class ReduceFinfoBase;
class ReduceBase;

typedef struct {
	ObjId oi;
	FuncId fid;
	unsigned int entrySize;
	unsigned int numEntries;
} ObjFid;

/**
 * This class manages information going into and out of the async queue.
 */
class Qinfo
{
	friend void testSendSpike();
	friend void testSparseMsg();
	public:
		/**
		 * Constructor.
		 * Used in addToQ and addDirectToQ.
		 * The size argument is in doubles, not bytes.
		 */
		Qinfo( const ObjId& src, 
			BindIndex bindIndex, ThreadId threadNum,
			unsigned int dataIndex, unsigned int dataSize );

		Qinfo();

		/// Used in readBuf to create a copy of Qinfo with specified thread.
		Qinfo( const Qinfo* other, ThreadId threadNum );

		//////////////////////////////////////////////////////////////
		// local Qinfo field access functions.
		//////////////////////////////////////////////////////////////

		/**
		 * Returns true if the Qinfo is inserted just for padding, and
		 * the data is not meant to be processed.
		 */
		bool isDummy() const {
			return dataIndex_ == 0;
		}

		/*
		MsgId mid() const {
			return m_;
		}
		*/

		bool isDirect() const {
			return msgBindIndex_ == DirectAdd;
		}

		/**
		 * Returns the thread index.
		 */
		ThreadId threadNum() const {
			return threadNum_;
		}

		/**
		 * Assigns the thread index
		 */
		void setThreadNum( ThreadId threadNum ) {
			threadNum_ = threadNum;
		}

		/**
		 * Returns the src ObjId
		 */
		ObjId src() const {
			return src_;
		}

		/**
		 * Returns the BindIndex
		 */
		BindIndex bindIndex() const {
			return msgBindIndex_;
		}

		/**
		 * Returns the index of the data associated with this Q entry.
		 * If it is the outQ, this refers to the location in 
		 * dBuf_[threadNum].
		 * If it is the inQ, this refers to the location in the array of
		 * doubles in the inQ.
		 */
		unsigned int dataIndex() const {
			return dataIndex_;
		}

		/**
		 * Lock Mutex, but only if a bunch of conditions are met:
		 * First we should be running pthreads
		 * Second, the system should be in multithread mode
		 * Third, must be running on Script thread.
		 * I put this into a function so that I can apply these 
		 * conditions whenver we have to lock the thread.
		 */
		static void qLockMutex();

		/**
		 * Unlock Mutex, but only if a bunch of conditions are met.
		 */
		static void qUnlockMutex();


		//////////////////////////////////////////////////////////////
		// Functions to put data on Q.
		//////////////////////////////////////////////////////////////
		/**
		 * Add data to the queue. Fills up an entry in the qBuf as well
		 * as putting the corresponding data.
		 */
		static void addToQ( const ObjId& oi, 
			BindIndex bindIndex, ThreadId threadNum,
			const double* arg, unsigned int size );

		static void addToQ( const ObjId& oi, 
			BindIndex bindIndex, ThreadId threadNum,
			const double* arg1, unsigned int size1,
			const double* arg2, unsigned int size2 );


		/**
		 * Add data to queue without any underlying message. This is used
		 * for specific point-to-point data delivery.
		 */
		static void addDirectToQ( const ObjId& src, const ObjId& dest,
			ThreadId threadNum,
			FuncId fid,
			const double* arg, unsigned int size );

		static void addDirectToQ( const ObjId& src, const ObjId& dest,
			ThreadId threadNum,
			FuncId fid,
			const double* arg1, unsigned int size1,
			const double* arg2, unsigned int size2 );


		/**
		 * Add vector data to queue without any underlying message,
		 * allocate space for the arguments and return a pointer to this 
		 * space. This function is used
		 * for specific point-to-point data delivery with a vector
		 */
		static void addVecDirectToQ( const ObjId& src, const ObjId& dest,
			ThreadId threadNum, FuncId fid,
			const double* arg, 
			unsigned int entrySize, unsigned int numEntries );


		/**
		 * Adds an existing queue entry into the structuralQ, for later
		 * execution when it is safe to do so.
		 * This is now thread-safe, has a mutex in it.
		 * Returns true if it added the entry.
		 * Returns false if it was in the Qinfo::clearStructuralQ function
		 * and wants the calling function to actually operate on the queue.
		 */
		bool addToStructuralQ() const;

		/**
		 * Locks Qmutex if multithreaded and on thread 0
		 */
		static void lockQmutex( ThreadId threadNum );

		/**
		 * Unlocks Qmutex if multithreaded and on thread 0
		 */
		static void unlockQmutex( ThreadId threadNum );

		//////////////////////////////////////////////////////////////
		// From here, static funcs handling the Queues.
		//////////////////////////////////////////////////////////////

		/**
		 * Read the inQ. Meant to run on all the sim threads.
		 * The Messages internally ensure thread safety by segregating
		 * target Objects.
		 */
		static void readQ( ThreadId threadNum );

		/**
		 * Move pending data in the qBuf and dBuf into the inQ.
		 * Must be protected by a mutex as it affects data on
		 * all threads.
		 */
		static void swapQ();

		/**
		 * Exchange mpiInQ and mpiRecvQ.
		 * Must be protected by a mutex as it affects data on
		 * all threads.
		 */
		static void swapMpiQ();

		/**
		 * Clears out contents of all qs, correspondingly the qBlock.
		 */
		static void emptyAllQs();

		/**
		 * Reporting function to tell us about queue status.
		 */
		static void reportQ();


		/**
		 * Used to work through the queues when the background
		 * threads are not running.
		 */
		static void clearQ( ThreadId threadNum );

		/**
		 * Returns a pointer to the data buffer in the specified inQ.
		 * Used for taking queue info to process.
		 * This points to different places, typically q0_[0] during
		 * single-node operation, and to mpiQ0 and mpiQ1 during 
		 * multinode function.
		 * The first entry in inQ is size (in doubles)
		 * The second entry in inQ is numQinfo
		 * The next set of entries are Qinfos
		 * The final set of entries, indexed from the Qinfos, are the data
		static const double* inQ();
		 */

		/**
		 * Returns a pointer to the data buffer on q0_[0].
		 * This is the data that is to go out from current node.
		 * Cannot be a const because the MPI call expects a variable.
		 * Same structure as inQ:
		 * The first entry in inQ is size (in doubles)
		 * The second entry in inQ is numQinfo
		 * The next set of entries are Qinfos
		 * The final set of entries, indexed from the Qinfos, are the data
		 */
		static double* sendQ();

		/**
		 * Returns a pointer to the block of memory on the mpiRecvQ.
		 */
		static double* mpiRecvQ();

		/**
		 * Works through any requests for structural changes to the model.
		 * This includes creation, deletion,
		 * resizing, and movement of Elements and Msgs. These functions
		 * must be carried out at a time when nothing else is being
		 * computed, and no iterators are pending. Currently happens during
		 * swapQ, and is serial and single-threaded.
		 */
		static void clearStructuralQ();

		/**
		 * Does an initial allocation of space for data transfer in 
		 * the qBuf and dBuf.
		 */
		static void initQs( unsigned int numThreads, unsigned int reserve );

		///////////////////////////////////////////////////////////////////
		// Here we have several functions that deal with the MPI data stuff.
		///////////////////////////////////////////////////////////////////
		/**
		 * updateQhistory() keeps track of recent bufsizes and has a simple 
		 * heuristic to judge how big the next mpi data transfer should be:
		 * It takes the second-biggest of the last historySize_ entries, 
		 * adds 10% and adds 10 to that.
		 * static func.
		 */

		static void updateQhistory();

		/**
		 * If an MPI packet comes in asking for more space than the 
		 * blockSize, this function resizes the buffer to fit.
		 */
		static void expandMpiRecvQ( unsigned int size );

		/**
		 * Assigns the number of the current source node for MPI data
		 */
		static void setSourceNode( unsigned int n );

		/**
		 * Returns the block size for data transfer from specified node.
		 */
		static unsigned int blockSize( unsigned int node );

		///////////////////////////////////////////////////////////////////
		// ReduceQ stuff.
		///////////////////////////////////////////////////////////////////
		/**
		 * Adds an entry to the ReduceQ. If the Eref is a new one it 
		 * creates a new Q entry with slots for each thread, otherwise
		 * just fills in the appropriate thread on an existing entry.
		 * I expect that these will be quite rare.
		 */
		static void addToReduceQ( ReduceBase* r, unsigned int threadIndex );

		/**
		 * Marches through reduceQ executing pending operations and finally
		 * freeing the Reduce entries and zeroing out the queue.
		 */
		static void clearReduceQ( unsigned int numThreads );

		///////////////////////////////////////////////////////////////////
		/**
		 * Zeroes the isSafeForStructuralOps_ flag.
		 */
		static void disableStructuralOps();

		/**
		 * Sets the isSafeForStructuralOps_ flag.
		 */
		static void enableStructuralOps();

		/**
		 * Blocks for the specified number of Process cycles, if the
		 * process loops and threads are operational. 
		 * Otherwise goes through the specified number of ClearQs.
		 */
		static void waitProcCycles( unsigned int numCyclesToWait );
	 

		/**
		 * Initializes the qMutex
		 */
		static void initMutex();

		/**
		 * Cleans up the qMutex
		 */
		static void freeMutex();

		bool execThread( Id id, unsigned int dataIndex ) const;

	private:
		ObjId src_;	/// Originating object

		BindIndex msgBindIndex_; /// Index to identify Msg and Fid on src
			/// Takes value of -1 when it is a DirectQentry.

		// ProcId proc_; /// Identifier for Process handled in Q.
		ThreadId threadNum_; /// Which thread am I on?

		/**
		 * Index to look up data, counting from start of array of 
		 * doubles in inQ.
		 * Index is zero if and only if this Qinfo is a dummy. Otherwise
		 * it points somewhere into the array.
		 */
		unsigned int dataIndex_;	

		/**
		 * Size of data packet on Qinfo. This is usually redundant info 
		 * to dataIndex, except when we need to addToStructuralQ.
		 * This is expressed in doubles, not in bytes.
		 */
		unsigned int dataSize_;

		///////////////////////////////////////////////////////////
		/**
		 * Organization of inQ_, which is used as input to each thread and
		 * is also transferred to other nodes:
		 * [0]: Size, that is, 1 + index of last occupied entry.
		 * [1]: Num Qinfo
		 * [2]: Qinfo[0] 	Arranged in blocks. This is Thread 0, index 0.
		 * [5]: Qinfo[1]	Thread 0, index 1
		 * [8]: Qinfo[2]	Thread 0, index 2
		 * ...
		 * [2+3*numQinfoOnThread0]: Thread 1, index 0
		 * ...
		 * [data00]: Index of zero data entry, belongs to thread 0.
		 * [data01]: Index of one data entry, belongs to thread 0.
		 * ...
		 * [data10]: Index of zero data entry, belongs to thread 1.
		 * ...
		 */
		static double* inQ_;	/// Data is read from the inQ
		static vector< unsigned int > qinfoCount_; ///# of qs on each thread

		/// Where to put nextQinfo, for each thread.
		static vector< unsigned int > nextQinfoIndex_; 
		/// Where to put nextData, for each thread.
		static vector< unsigned int > nextDataIndex_; 

		static vector< double > q0_;	/// Allocated space for data in q

		/**
		 * The data buffer. One vector per thread. Data is written here
		 * as the thread calculations proceed.
		 */
		static vector< vector< double > > dBuf_;	

		/**
		 * The Qinfo buffer. One vector per thread. Qinfos are written here
		 * as the thread calculations proceed. The Qinfos have a dataIndex 
		 * that specifies the location in the dBuf into which the data is
		 * going. To get the size you need to look at the next qBuf entry.
		 */
		static vector< vector< Qinfo > > qBuf_;

		/**
		 * Ugly flag to tell Shell functions if the simulation should
		 * actually compute structural operations, or if it should just
		 * stuff them into a buffer.
		 */
		static bool isSafeForStructuralOps_;

		/*
		 * This handles incoming data from MPI. It is used as a buffer
		 * for the MPI_Bcast or other calls to dump internode data into.
		 * Currently the outgoing data is sent each timestep from the inQ.
		 */
		static double* mpiRecvQ_;

		/**
		 * These are the actual allocated locations of the vectors
		 * underlying the mpiRecvQ and mpiInQ.
		 * We only need one Qvec for each, since at any given time one
		 * node will be sending in data and we'll be processing the
		 * arrived data from the previous node. Note that we send data
		 * one node at a time, and if the current node is the source of
		 * more than one group, it sends only one group at a time.
		 * Once the data is in, it does not care about originating
		 * node/group since all threads chew on it anyway.
		 */
		static vector< double > mpiQ0_;
		static vector< double > mpiQ1_;

		/**
		 * This contains pointers to Queue entries requesting functions that
		 * change the model structure.
		 * We only need to point to the original Queue entry, because it
		 * sits safely in the InQ. Likewise, its data is unchanged sitting
		 * in the InQ. 
		 * Operations using the structuralQ include creation, deletion,
		 * resizing, and movement of Elements and Msgs. These functions
		 * must be carried out at a time when nothing else is being
		 * computed, and no iterators are pending. Currently happens during
		 * swapQ, and is serial and single-threaded.
		 */
		// static vector< double > structuralQ_;
		static vector< Qinfo > structuralQinfo_;
		static vector< double > structuralQdata_;

		/**
		 * The reduceQ manages requests to 'reduce' data from many sources.
		 * This Q has to keep track of running totals on each thread, then
		 * it digests them across threads and finally across nodes.
		 * The initial running total begins at phase2/3 of the process 
		 * loop, on many threads. The final summation is done in barrier3.
		 * After barrier3 the reduceQ_ should be empty.
		 */
		static vector< vector< ReduceBase* > > reduceQ_;

		/**
		 * Used to protect the queues from the timing of parser calls
		 */
		static pthread_mutex_t* qMutex_;
		static pthread_cond_t* qCond_;

		static bool waiting_;

		/// Number of cycles to wait before returning in Set/Get.
		static int numCyclesToWait_;

		/// Number of cycles completed since the ProcessLoop started
		static unsigned long numProcessCycles_;

		/// The # of the source node being handled for MPI data transfers.
		static unsigned int sourceNode_;

		/// Flag setting
		static const BindIndex DirectAdd;

		/// Extra Margin (fraction) to leave for blocksize
		static const double blockMargin_;

		/// Number of entries of buffer size to retain for MPI data transfer
		static const unsigned int historySize_;

		/// History of recent buffer transfers using MPI.
		static vector< vector< unsigned int > > history_;

		/// Size of data blocks to send to each node
		static vector< unsigned int > blockSize_;
};

#endif // QINFO_H
