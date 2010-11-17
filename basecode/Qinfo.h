/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

/// Forward declaration of Qvec.
class Qvec;

/**
 * This little class keeps track of blocks of data destined for different
 * queues.
 */
class QueueBlock {
	public:
		QueueBlock()
			: whichQ( 0), startOffset( 0 ), size( 0 )
		{;}

		QueueBlock( short wq, unsigned int so, unsigned int si )
			: whichQ( wq), startOffset( so ), size( si )
		{;}
		short whichQ;
		unsigned int startOffset; // Position of data wrt start of Q.
		unsigned int size; // size in bytes
};

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
		 */
		static Qinfo makeDummy( unsigned int size );

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
		 * Returns true if the Qinfo is inserted just for padding, and
		 * the data is not meant to be processed.
		 */
		bool isDummy() const {
			return isDummy_;
		}

		/*
		void setForward( bool isForward ) {
			isForward_ = isForward;
		}
		*/

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
		 * Read the inQ. Meant to run on all the sim threads.
		 * The Messages internally ensure thread safety by segregating
		 * target Objects.
		 */
		static void readQ( const ProcInfo* proc );

		/**
		 * Read the MPI Q. Similar to readQ, except that the data source
		 * has arrived from off-node.
		 */
		static void readMpiQ( const ProcInfo* proc );

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
		 * Send contents of root inQ to all nodes using MPI, gather
		 * their pending return functions.
		static void sendRootToAll( const ProcInfo* proc );
		 */

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
		 * Utility function that checks if Msg is local, and if so,
		 * assigns Qblock and adds the data to a suitable Queue.
		void assembleOntoQ( const MsgFuncBinding &b, 
			const Element* e, const ProcInfo* p, const char* arg );
		 */
	
		/**
		 * Organizes data going into outQ so that we know which
		 * execution queue the data is due to end up in, and how big
		 * each block is to be.
		void assignQblock( const Msg* m, const ProcInfo* p );
		 */

		/**
		 * Used to work through the queues when the background
		 * threads are not running.
		 */
		static void clearQ( const ProcInfo* p );
		static void mpiClearQ( const ProcInfo* p );

		/**
		 * Works through any requests for structural changes to the model.
		 * This includes creation, deletion,
		 * resizing, and movement of Elements and Msgs. These functions
		 * must be carried out at a time when nothing else is being
		 * computed, and no iterators are pending. Currently happens during
		 * swapQ, and is serial and single-threaded.
		 */
		static void clearStructuralQ();

	private:
		bool useSendTo_;	/// true if msg is to a single target DataId.
		bool isForward_; /// True if the msg is from e1 to e2.
		bool isDummy_; /// True if the Queue entry is a dummy and not used.
		MsgId m_;		/// Unique lookup Id for Msg.
		FuncId f_;		/// Unique lookup Id for function.
		DataId srcIndex_; /// DataId of src.
		unsigned int size_; /// size of argument in bytes. Zero is allowed.

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
		 * This contains pointers to Queue entries requesting functions that
		 * change the model structure. This includes creation, deletion,
		 * resizing, and movement of Elements and Msgs. These functions
		 * must be carried out at a time when nothing else is being
		 * computed, and no iterators are pending. Currently happens during
		 * swapQ, and is serial and single-threaded.
		 */
		static vector< const char* > structuralQ_;

		/**
		 * This handles incoming data from MPI. At this point it
		 * is not meant to be filled locally, but if we move to
		 * a system of staggered Bcast sends then we may need to
		 * maintain mpiInQ and mpiOutQ. Currently the outgoing
		 * data is sent each timestep from the inQ.
		 */
		static vector< Qvec > mpiQ_;
		
		/**
		 * This keeps track of which data go into which queue.
		 * This accompanies each outQ. At the time messages are dumped
		 * into outQ, the Msgs need to assign suitable queues.
		 * Each Qblock has start, size, and target queue.
		 */
		static vector< vector< QueueBlock > > qBlock_;

		static vector< SimGroup > g_; // Information about grouping.
};
