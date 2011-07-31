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
		/// Used in addToQ and addDirectToQ
		Qinfo( const ObjId& src, 
			BindIndex bindIndex, unsigned short threadNum,
			unsigned int size );

		Qinfo();

		/*
		void setMsgId( MsgId m ) {
			m_ = m;
		}
		*/

		/**
		 * Returns true if the data is to go to a specific one among
		 * all the message targets. 
		bool useSendTo() const {
			return useSendTo_;
		}
		 */

		/**
		 * Returns true if the direction of the message is from
		 * e1 to e2.
		bool isForward() const {
			return isForward_;
		}
		 */

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
			return msgBindIndex_ == ~0;
		}

		/**
		 * Returns the thread index.
		 */
		unsigned short threadNum() const {
			return threadNum_;
		}

		void setThreadNum( unsigned short threadNum ) {
			threadNum_ = threadNum;
		}

		/*
		void setFid( FuncId f ) {
			f_ = f;
		}
		*/

		/*
		DataId srcIndex() const {
			return srcIndex_;
		}
		*/
		ObjId src() const {
			return src_;
		}

		BindIndex bindIndex() const {
			return msgBindIndex_;
		}

		unsigned int dataIndex() const {
			return dataIndex_;
		}

		/**
		 * size() returns the length of the data segment managed by this 
		 * Qinfo, and immediately following it. Note that the total
		 * length in memory of of this entire queue entry is 
		 * sizeof( Qinfo ) + Qinfo::size()
		unsigned int size() const {
			return size_;
		}
		 */

		/**
		 * Add data to the queue. Fills up an entry in the qBuf as well
		 * as putting the corresponding data.
		 */
		static void addToQ( const ObjId& oi, 
			BindIndex bindIndex, unsigned short threadNum,
			const double* arg, unsigned int size );

		/**
		 * Add data to the queue. Fills up an entry in the qBuf, allocate
		 * space for the arguments, and return a pointer to this space.
		 * Used for multiple args.
		 */
		static double* addToQ( const ObjId& oi, 
			BindIndex bindIndex, unsigned short threadNum,
			unsigned int size );

		/**
		 * Add data to queue without any underlying message. This is used
		 * for specific point-to-point data delivery.
		 */
		static void addDirectToQ( const ObjId& src, const ObjId& dest,
			unsigned short threadNum,
			FuncId fid,
			const double* arg, unsigned int size );

		/**
		 * Add data to queue without any underlying message, allocate space
		 * for the arguments and return a pointer to this space. 
		 * This is used
		 * for specific point-to-point data delivery with multiple args.
		 */
		static double* addDirectToQ( const ObjId& src, const ObjId& dest,
			unsigned short threadNum,
			FuncId fid,
			unsigned int size );

		/**
		 * Add vector data to queue without any underlying message,
		 * allocate space for the arguments and return a pointer to this 
		 * space. This function is used
		 * for specific point-to-point data delivery with a vector
		 */
		static double* addVecDirectToQ( const ObjId& src, const ObjId& dest,
			unsigned short threadNum,
			FuncId fid,
			unsigned int entrySize, unsigned int numEntries );


		/**
		 * Adds an existing queue entry into the structuralQ, for later
		 * execution when it is safe to do so.
		 * This is not thread-safe, should only be called by the Shell.
		 * Returns true if it added the entry.
		 * Returns false if it was in the Qinfo::clearStructuralQ function
		 * and wants the calling function to actually operate on the queue.
		 */
		bool addToStructuralQ( const double* data, unsigned int size) const;

		/**
		 * This adds the data to the queue and then an additional
		 * sizeof( DataId ) block to specify target DataId.
		 */
		void addSpecificTargetToQ( const ProcInfo* p, MsgFuncBinding b, 
			const char* arg, const DataId& target, bool isForward );


		/**
		 * This assigns temporary storage in the Qinfo for thread
		 * identifiers.
		void setProcInfo( const ProcInfo* p );
		 */

		/**
		 * This extracts the procinfo.
		const ProcInfo* getProcInfo() const;
		 */
		//////////////////////////////////////////////////////////////
		// From here, static funcs handling the Queues.
		//////////////////////////////////////////////////////////////

		/**
		 * Set up a SimGroup which keeps track of grouping information, and
		 * resulting queue information.
		 * Returns group#
		static unsigned int addSimGroup( unsigned short numThreads,
			unsigned short numNodes );
		 */

		/**
		 * 	Returns the number of SimGroups
		static unsigned int numSimGroup();
		 */

		/**
		 * Returns the specified SimGroup
		static const SimGroup* simGroup( unsigned int index );
		 */

		/**
		 * Clears out all sim groups.
		static void clearSimGroups();
		 */

		/**
		 * Read the inQ. Meant to run on all the sim threads.
		 * The Messages internally ensure thread safety by segregating
		 * target Objects.
		 */
		static void readQ( const ProcInfo* proc );

		/**
		 * Read the MPI Q. Similar to readQ, except that the data source
		 * has arrived from off-node.
		 */
		static void readMpiQ( const ProcInfo* proc, unsigned int node );

		/**
		 * Read the MPI Q in contexts where only the message from the
		 * root Element should be considered.
		static void readRootQ( const ProcInfo* proc );
		 */

		/**
		 * Exchange inQ and outQ.
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
		 * Send contents of specified inQ to all nodes using MPI
		 */
		static void sendAllToAll( const ProcInfo* proc );

		/**
		 * Handles the case where the system wants to send a msg to
		 * a single target. Currently done through an ugly hack, 
		 * encapsulated here.
		static void hackForSendTo( const Qinfo* q, const char* buf );
		 */

		/**
		 * Reporting function to tell us about queue status.
		 */
		static void reportQ();


		/**
		 * Used to work through the queues when the background
		 * threads are not running.
		 */
		static void clearQ( const ProcInfo* p );

		/**
		 * Dummy function, in due course will monitor data transfer
		 * stats and figure out the best data transfer size for each
		 * group as it goes along.
		 */
		static void doMpiStats( unsigned int bufsize, const ProcInfo* p );

		/**
		 * Returns a pointer to the data buffer in the specified inQ
		 * Cannot be a const because the MPI call expects a variable.
		 */
		static double* inQ();

		/**
		 * Looks up size of data buffer in specifid inQ
		static unsigned int inQdataSize( unsigned int group );
		 */

		/**
		 * Returns a pointer to the block of memory on the mpiRecvQ.
		 */
		static double* mpiRecvQbuf();

		/**
		 * Expands the memory allocation on mpiRecvQ to handle outsize
		 * data packets
		static void expandMpiRecvQbuf( unsigned int size );
		 */

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
		 * the MPIQs.
		 */
		static void initMpiQs();

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

		/**
		 * Sets the isSafeForStructuralOps_ flag.
		 * Helper function used by the async unit tests.
		 * Not to be used elsewhere.
		 */
		static void disableStructuralQ();

		/**
		 * Zeroes the isSafeForStructuralOps_ flag.
		 * Helper function used by the async unit tests.
		 * Not to be used elsewhere.
		 */
		static void enableStructuralQ();

	private:
		ObjId src_;	/// Originating object

		BindIndex msgBindIndex_; /// Index to identify Msg and Fid on src
			/// Takes value of -1 when it is a DirectQentry.

		// ProcId proc_; /// Identifier for Process handled in Q.
		unsigned short threadNum_; /// Which thread am I on?

		/**
		 * fid_ is zero for regular forward msgs.
		 * In reverse msgs, fid_ identifies target func. In these cases the
		 * msgBindIndex is the MsgId.
		 * Not needed here: if we need it we should in any case have the
		 * extra space in the Data portion of the queue entry, where the
		 * dest ObjId lives, to also put the FuncId.
		FuncId fid_; 				
		 */

		/**
		 * Index to look up data, counting from start of array of 
		 * doubles in inQ.
		 * Index is zero if and only if this Qinfo is a dummy. Otherwise
		 * it points somewhere into the array.
		 */
		unsigned int dataIndex_;	

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
		static double* outQ_;	/// Outgoing Msg data is written to outQ.
		static double* inQ_;	/// Data is read from the inQ
		static vector< unsigned int > qinfoCount_; ///# of qs on each thread

		/// Where to put nextQinfo, for each thread.
		static vector< unsigned int > nextQinfoIndex_; 
		/// Where to put nextData, for each thread.
		static vector< unsigned int > nextDataIndex_; 

		static vector< double > q0_;	/// Allocated space for data in q
		static vector< double > q1_;	/// Allocated space for data in q

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

		///////////////////////////////////////////////////////////

		/*
		bool useSendTo_;	/// true if msg is to a single target DataId.
		bool isForward_; /// True if the msg is from e1 to e2.

		// Deprecated
		// bool isDummy_; /// True if the Queue entry is a dummy and not used.

		MsgId m_;		/// Unique lookup Id for Msg.
		FuncId f_;		/// Unique lookup Id for function.
		DataId srcIndex_; /// DataId of src.
		unsigned int size_; /// size of argument in bytes. Zero is allowed.

		unsigned short procIndex_; /// Which thread does Q entry go to?
		///////////////////////////////////////////////////////////
		*/

		/**
		 * Ugly flag to tell Shell functions if the simulation should
		 * actually compute structural operations, or if it should just
		 * stuff them into a buffer.
		 */
		static bool isSafeForStructuralOps_;

		/**
		 * outQ_ is the buffer in which messages get queued. The Qvec
		 * class deals with threading issues.
		 * There are as many entries as there are simulation groups.
		 * In computation phase 2 the outQ swaps with the inQ, and the 
		 * inQ is used to read the data that had accumulated in the outQ.
		static vector< Qvec >* outQ_;
		 */

		/**
		 * inQ_ is the buffer that holds data to be read out in order to
		 * deliver the messages to the target.
		static vector< Qvec >* inQ_;
		 */

		/*
		 * This handles incoming data from MPI. It is used as a buffer
		 * for the MPI_Bcast or other calls to dump internode data into.
		 * Currently the outgoing data is sent each timestep from the inQ.
		static Qvec* mpiRecvQ_;
		 */
		static double* mpiRecvQ_;

		/**
		 * This handles data that has arrived from MPI on the previous
		 * transfer, and is now stable so it can be read through.
		 * Equivalent in this role to the inQ_.
		 */
		static double* mpiInQ_;

		/**
		 * These are the actual allocated locations of the vectors
		 * underlying the inQ and outQ.
		 * The number of entries in the vectors is equal to the number
		 * of simulation groups, which have close message coupling
		 * requiring all-to-all MPI communications.
		static vector< Qvec > q1_;
		static vector< Qvec > q2_;
		 */

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
		static Qvec mpiQ1_;
		static Qvec mpiQ2_;
		 */
		static vector< double > mpiQ1_;
		static vector< double > mpiQ2_;

		/**
		 * This contains pointers to Queue entries requesting functions that
		 * change the model structure. This includes creation, deletion,
		 * resizing, and movement of Elements and Msgs. These functions
		 * must be carried out at a time when nothing else is being
		 * computed, and no iterators are pending. Currently happens during
		 * swapQ, and is serial and single-threaded.
		 */
		// static vector< double > structuralQ_;
		static vector< Qinfo > structuralQinfo_;
		static vector< double > structuralQdata_;

		/*
		** Deprecated.
		static vector< SimGroup > g_; // Information about grouping.
		*/

		/**
		 * The reduceQ manages requests to 'reduce' data from many sources.
		 * This Q has to keep track of running totals on each thread, then
		 * it digests them across threads and finally across nodes.
		 * The initial running total begins at phase2/3 of the process 
		 * loop, on many threads. The final summation is done in barrier3.
		 * After barrier3 the reduceQ_ should be empty.
		 */
		static vector< vector< ReduceBase* > > reduceQ_;
};

#endif // QINFO_H
