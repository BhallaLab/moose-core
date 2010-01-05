/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

// The # of queues is around 2x the # of threads (including offnode ones)
// in the hardware. There do exist machines where a short will not suffice,
// but not too many of them at this time!
typedef unsigned short Qid;

class SimGroup {
	public: 
		SimGroup( unsigned short nt, unsigned short si )
			: numThreads( nt ), startIndex( si )
			{;}
		unsigned short numThreads; // Number of threads in this group.
		unsigned short startIndex; // Index of first thread, used for inQ.

		/**
		 * returns Qid for the thread specified within this group
		Qid outQ( unsiged int relativeThreadId ) const {
			assert( relativeThreadId < numThreads );
			return startIndex + threadId + 1;
		}
		 */

		/**
		 * returns Qid for the absolute threadId
		 */
		Qid outQ( unsigned int threadId, unsigned int groupIndex ) const {
			Qid ret = threadId + groupIndex + 1;
			assert( ret - startIndex < numThreads );
			return( ret );
		}
		// Stuff here for off-node queues.
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

		void setForward( bool isForward ) {
			isForward_ = isForward;
		}

		MsgId mid() const {
			return m_;
		}

		FuncId fid() const {
			return f_;
		}

		DataId srcIndex() const {
			return srcIndex_;
		}

		unsigned int size() const {
			return size_;
		}

		/**
		 * Used when adding space for the index in sendTo
		 */
		void expandSize();

		/**
		 * Decide how many queues to use, and their reserve size
		 * Deprecated
		 */
		// static void setNumQs( unsigned int n, unsigned int reserve );

		/**
		 * Set up a SimGroup which keeps track of grouping information, and
		 * resulting queue information.
		 * Returns group#
		 */
		static unsigned int addSimGroup( unsigned short numThreads );

		/**
		 * Legacy utility function, just a readQ followed by zeroQ.
		 */
		static void clearQ( Qid qId );

		/**
		 * Read the queue specified by the ProcInfo. Depending on the
		 * scheduling and threading structure, may simply go through
		 * all the available queues.
		 */
		static void readQ( Qid qId );

		/**
		 * Zeroes out contents (or simply resizes) queue
		 */
		static void zeroQ( Qid qId );

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
		 * Add data to the queue. This is non-static, since we will also
		 * put the current Qinfo on the queue as a header.
		 * The arg will just be memcopied onto the queue, so avoid
		 * pointers. Possibly add size as an argument
		 */
		void addToQ( Qid qId, MsgId mid, bool isForward, const char* arg );

	private:
		MsgId m_;
		bool useSendTo_;	// true if the msg is to a single target DataId.
		bool isForward_; // True if the msg is from e1 to e2.
		FuncId f_;
		DataId srcIndex_; // DataId of src.
		unsigned int size_; // size of argument in bytes.
		static vector< vector< char > > q_; // Here are the queues
		static vector< SimGroup > g_; // Information about grouping.
};
