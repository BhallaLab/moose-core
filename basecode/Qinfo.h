/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

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
		Qinfo( FuncId f, DataId srcIndex, unsigned int size );

		Qinfo( bool useSendTo, bool isForward,
			DataId srcIndex, unsigned int size );

		Qinfo( FuncId f, DataId srcIndex, 
			unsigned int size, bool useSendTo, bool isForward );

		Qinfo();
		// Qinfo( const char* buf );

		void setMsgId( MsgId m ) {
			m_ = m;
		}

		bool useSendTo() const {
			return useSendTo_;
		}

		bool isForward() const {
			return isForward_;
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

		unsigned int size() const {
			return size_;
		}


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
		 * Legacy utility function. The proc must specify the correct
		 * group. clearQ merges all outQs in the group into its inQ,
		 * then reads the inQ, and then clears the inQ.
		 */
		static void clearQ( const ProcInfo *proc );

		/**
		 * Variant that also takes care of internode stuff.
		 */
		static void mpiClearQ( const ProcInfo *proc );

		/**
		 * Read the inQ specified by the ProcInfo.
		 */
		static void readQ( const ProcInfo* proc );

		/**
		 * Read the localQ.
		 */
		static void readLocalQ( const ProcInfo* proc );

		/**
		 * Read the MPI Q
		 */
		static void readMpiQ( const ProcInfo* proc );

		/**
		 * Merge all outQs from a group into its inQ.
		 */
		static void mergeQ( unsigned int groupId );

		/**
		 * Load a buffer of data into an inQ. Assumes threading has been
		 * dealt with.
		static void loadQ( Qid qId, const char* buf, unsigned int length );
		 */

		/**
		 * Dump an inQ into a buffer of data. Again, assumes threading has
		 * been dealt with. Basically a memcpy.
		static unsigned int dumpQ( Qid qid, char* buf );
		 */

		/**
		 * Send contents of specified inQ to all nodes using MPI
		 */
		static void sendAllToAll( const ProcInfo* proc );

		/**
		 * Send contents of root inQ to all nodes using MPI, gather
		 * their pending return functions.
		 */
		static void sendRootToAll( const ProcInfo* proc );

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
		 * Add data to the queue. This is non-static, since we will also
		 * put the current Qinfo on the queue as a header.
		 * The arg will just be memcopied onto the queue, so avoid
		 * pointers. Possibly add size as an argument
		 */
		void addToQ( unsigned int threadId, MsgFuncBinding b, 
			const char* arg );
		void addSpecificTargetToQ( unsigned int threadId, MsgFuncBinding b, 
			const char* arg, const DataId& target );
	
		/**
		 * Organizes data going into outQ so that we know which
		 * execution queue the data is due to end up in, and how big
		 * each block is to be.
		 */
		void assignQblock( const Msg* m, const ProcInfo* p );
		
	private:
		bool useSendTo_;	// true if the msg is to a single target DataId.
		bool isForward_; // True if the msg is from e1 to e2.
		MsgId m_;
		FuncId f_;
		DataId srcIndex_; // DataId of src.
		unsigned int size_; // size of argument in bytes.

		/**
		 * outQ is one per worker thread. The immediate output goes into
		 * the outQs which are later consolidated.
		 */
		static vector< vector< char > > outQ_;

		/**
		 * inQ is one per SimGroup. It becomes a readonly vector once
		 * consolidated, and all the threads in the group read from it.
		 * Each inQ has a header of sizeof( unsigned int ) that contains
		 * the buffer size, in bytes. This size INCLUDES the header.
		 */
		static vector< vector< char > > inQ_;

		/**
		 * There are numCores mpiQ blocks per SimGroup, but the blocks
		 * for each SimGroup are arranged as one long linear array.
		 */
		static vector< vector< char > > mpiQ_;

		/**
		 * This is a single, simple queue that lives only on the local node.
		 * It is for messages that are not going even to other elements
		 * in the same SimGroup.
		 * Examples are SetGet messages, and messages to globals.
		 * It is populated by examining outQ for local-only messages.
		 */
		static vector< char > localQ_;

		/**
		 * This keeps track of which data go into which queue.
		 * This accompanies each outQ. At the time messages are dumped
		 * into outQ, the Msgs need to assign suitable queues.
		 * Each Qblock has start, size, and target queue.
		 */
		static vector< vector< QueueBlock > > qBlock_;

		static vector< SimGroup > g_; // Information about grouping.
};
