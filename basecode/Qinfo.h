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

/**
 * This class manages information going into and out of the async queue.
 */
class Qinfo
{
	friend void testSendSpike();
	public:
		Qinfo( FuncId f, DataId srcIndex, unsigned int size );

		Qinfo( FuncId f, DataId srcIndex, 
			unsigned int size, bool useSendTo, bool isForward );

		Qinfo( const char* buf );

		void setMsgId( MsgId m ) {
			m_ = m;
		}

		// void addToQ( vector< char >& q_, const char* arg ) const;

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
		 */
		static void setNumQs( unsigned int n, unsigned int reserve );

		/**
		 * Clear the specified queueueueue.
		 */
		static void clearQ( Qid qId );

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
		// void addToQ( Qid qId, const char* arg );
		void addToQ( Qid qId, MsgId mid, bool isForward, const char* arg );

	private:
		MsgId m_;
		bool useSendTo_;	// true if the msg is to a single target DataId.
		bool isForward_; // True if the msg is from e1 to e2.
		FuncId f_;
		DataId srcIndex_; // DataId of src.
		unsigned int size_; // size of argument in bytes.
		static vector< vector< char > > q_; // Here are the queues
};
