/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

/// Forward declarations
class Qvec;
class ReduceFinfoBase;
class ReduceBase;


/**
 * This class manages information going into and out of the async queue.
 */
class Qinfo
{
	friend void testSendSpike();
	friend void testSparseMsg();
	public:
		// Qinfo( FuncId f, DataId srcIndex, unsigned int size );

		Qinfo( DataId srcIndex, unsigned int size, bool useSendTo );

		Qinfo( FuncId f, DataId srcIndex, 
			unsigned int size, bool useSendTo );

		Qinfo();
		// Qinfo( const char* buf );

		/**
		 * This is a special constructor to return dummy Qs.
		 * Only used in Qvec::stitch.
		 * Deprecated
		static Qinfo makeDummy( unsigned int size );
		 */

		void setMsgId( MsgId m ) {
			m_ = m;
		}

		/**
		 * Returns true if the data is to go to a specific one among
		 * all the message targets. 
		 */
		bool useSendTo() const {
			return useSendTo_;
		}

		/**
		 * Returns true if the direction of the message is from
		 * e1 to e2.
		 */
		bool isForward() const {
			return isForward_;
		}

		/**
		 * Deprecated
		 * Returns true if the Qinfo is inserted just for padding, and
		 * the data is not meant to be processed.
		 */
		bool isDummy() const {
			return 0;
			// return isDummy_;
		}

		MsgId mid() const {
			return m_;
		}

		FuncId fid() const {
			return f_;
		}

		/*
		void setFid( FuncId f ) {
			f_ = f;
		}
		*/

		DataId srcIndex() const {
			return srcIndex_;
		}

		/**
		 * size() returns the length of the data segment managed by this 
		 * Qinfo, and immediately following it. Note that the total
		 * length in memory of of this entire queue entry is 
		 * sizeof( Qinfo ) + Qinfo::size()
		 */
		unsigned int size() const {
			return size_;
		}

		/**
		 * Add data to the queue. This is non-static, since we will also
		 * put the current Qinfo on the queue as a header.
		 * The arg will just be memcopied onto the queue, so avoid
		 * pointers. The Qinfo must already know its size.
		 * This variant sets the isForward flag to True.
		 */
		void addToQforward( const ProcInfo* p, MsgFuncBinding b, const char* arg );

		/**
		 * Adds data to the queue with the isForward flag set to False.
		 */
		void addToQbackward( const ProcInfo* p, MsgFuncBinding b, const char* arg );

		/**
		 * Adds an existing queue entry into the structuralQ, for later
		 * execution when it is safe to do so.
		 * This is not thread-safe, should only be called by the Shell.
		 * Returns true if it added the entry.
		 * Returns false if it was in the Qinfo::clearStructuralQ function
		 * and wants the calling function to actually operate on the queue.
		 */
		bool addToStructuralQ() const;

		/**
		 * This adds the data to the queue and then an additional
		 * sizeof( DataId ) block to specify target DataId.
		 */
		void addSpecificTargetToQ( const ProcInfo* p, MsgFuncBinding b, 
			const char* arg, const DataId& target, bool isForward );


		/**
		 * This assigns temporary storage in the Qinfo for thread
		 * identifiers.
		 */
		void setProcInfo( const ProcInfo* p );

		/**
		 * This extracts the procinfo.
		 */
		const ProcInfo* getProcInfo() const;
		//////////////////////////////////////////////////////////////
		// From here, static funcs handling the Queues.
		//////////////////////////////////////////////////////////////

		/**
		 * Set up a SimGroup which keeps track of grouping information, and
		 * resulting queue information.
		 * Returns group#
		 */
		static unsigned int addSimGroup( unsigned short numThreads,
			unsigned short numNodes );

		/**
		 * 	Returns the number of SimGroups
		 */
		static unsigned int numSimGroup();

		/**
		 * Returns the specified SimGroup
		 */
		static const SimGroup* simGroup( unsigned int index );

		/**
		 * Clears out all sim groups.
		 */
		static void clearSimGroups();

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
		static char* inQ( unsigned int group );

		/**
		 * Looks up size of data buffer in specifid inQ
		 */
		static unsigned int inQdataSize( unsigned int group );

		/**
		 * Returns a pointer to the block of memory on the mpiRecvQ.
		 */
		static char* mpiRecvQbuf();

		/**
		 * Expands the memory allocation on mpiRecvQ to handle outsize
		 * data packets
		 */
		static void expandMpiRecvQbuf( unsigned int size );

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
		bool useSendTo_;	/// true if msg is to a single target DataId.
		bool isForward_; /// True if the msg is from e1 to e2.

		// Deprecated
		// bool isDummy_; /// True if the Queue entry is a dummy and not used.

		MsgId m_;		/// Unique lookup Id for Msg.
		FuncId f_;		/// Unique lookup Id for function.
		DataId srcIndex_; /// DataId of src.
		unsigned int size_; /// size of argument in bytes. Zero is allowed.

		unsigned short procIndex_; /// Which thread does Q entry go to?

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
		 */
		static vector< Qvec >* outQ_;

		/**
		 * inQ_ is the buffer that holds data to be read out in order to
		 * deliver the messages to the target.
		 */
		static vector< Qvec >* inQ_;

		/*
		 * This handles incoming data from MPI. It is used as a buffer
		 * for the MPI_Bcast or other calls to dump internode data into.
		 * Currently the outgoing data is sent each timestep from the inQ.
		 */
		static Qvec* mpiRecvQ_;

		/**
		 * This handles data that has arrived from MPI on the previous
		 * transfer, and is now stable so it can be read through.
		 * Equivalent in this role to the inQ_.
		 */
		static Qvec* mpiInQ_;

		/**
		 * These are the actual allocated locations of the vectors
		 * underlying the inQ and outQ.
		 * The number of entries in the vectors is equal to the number
		 * of simulation groups, which have close message coupling
		 * requiring all-to-all MPI communications.
		 */
		static vector< Qvec > q1_;
		static vector< Qvec > q2_;

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
		static Qvec mpiQ1_;
		static Qvec mpiQ2_;

		/**
		 * This contains pointers to Queue entries requesting functions that
		 * change the model structure. This includes creation, deletion,
		 * resizing, and movement of Elements and Msgs. These functions
		 * must be carried out at a time when nothing else is being
		 * computed, and no iterators are pending. Currently happens during
		 * swapQ, and is serial and single-threaded.
		 */
		static vector< char > structuralQ_;

		static vector< SimGroup > g_; // Information about grouping.

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
