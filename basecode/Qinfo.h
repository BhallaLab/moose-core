/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

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
		static unsigned int addSimGroup( unsigned short numThreads );

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
		 * Read the queue specified by the ProcInfo. Depending on the
		 * scheduling and threading structure, may simply go through
		 * all the available queues.
		 */
		static void readQ( const ProcInfo* proc );

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
		 */
		static void loadQ( Qid qId, const char* buf, unsigned int length );

		/**
		 * Dump an inQ into a buffer of data. Again, assumes threading has
		 * been dealt with. Basically a memcpy.
		 */
		static unsigned int dumpQ( Qid qid, char* buf );

		/**
		 * Handles the case where the system wants to send a msg to
		 * a single target. Currently done through an ugly hack, 
		 * encapsulated here.
		 */
		static void hackForSendTo( const Qinfo* q, const char* buf );

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
		// void addToQ( Qid qId, MsgId mid, bool isForward, const char* arg );
		void addToQ( Qid qId, MsgFuncBinding b, const char* arg );
		void addSpecificTargetToQ( Qid qId, MsgFuncBinding b, 
			const char* arg, const DataId& target );
	
		/**
		 * Returns a pointer to the inQ. Readonly.
		 */
		const char* getInQ( unsigned int i );

		/**
		 * Returns a pointer to the first block of the mpiQ, which is one
		 * long array.
		 * Note that the mpiQ has as many blocks as there are nodes,
		 * including current one. All blocks are the same size.
		 */
		char* getMpiQ( unsigned int i );
		
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
		 */
		static vector< vector< char > > inQ_;

		/**
		 * There are numCores mpiQ blocks per SimGroup, but the blocks
		 * for each SimGroup are arranged as one long linear array.
		 */
		static vector< vector< char > > mpiQ_;

		static vector< SimGroup > g_; // Information about grouping.
};
