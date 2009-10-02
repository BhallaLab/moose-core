/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2009 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

class Qinfo;
/**
 * Have a single Element class, that uses an IndexRange. There can
 * be multiple such objects representing a complete array. So the Id
 * must know how to access them all. Or they each refer to each other.
 * Or the user only sees the master (but may be many if distrib over nodes)
 */
class Element
{
	friend void testSync();
	friend void testAsync();
	friend void testStandaloneIntFire();
	friend void testSynapse();
	friend void testSyncArray( unsigned int, unsigned int, unsigned int );
	public:
		/**
		 * Constructor
		 * It would be nice to have shared data: e.g., thresh and tau for
		 * IntFire. Common but not static.
		 * Also think about parent-child hierarchy.
		 */
		/*
		Element( vector< Data * >& d, 
			unsigned int numSendSlots_,
			unsigned int numRecvSlots_
		);
		*/
		Element( const Cinfo* c, 
			Data* d, unsigned int numData, unsigned int dataSize );

		/**
		 * Destructor
		 */
		~Element();

		/**
		 * Here we build the array on the fly.
		Element( const Data *prototype, unsigned int numEntries );
		 */

		/**
		 * Examine process queue, data that is expected every timestep.
		 * This function is done for all the local data entries in order.
		 */
		void process( const ProcInfo* p );

		/**
		 * Clear sporadic message queue, events that come randomly.
		 */
		void clearQ();

		/**
		 * execFunc executes the function defined in the buffer, and
		 * returns the next position of the buffer. Returns 0 at the
		 * end.
		 */
		const char* execFunc( const char* buf );

		/**
		 * Return a single buffer entry specified by slot and eindex
		 */
		double oneBuf( SyncId slot, unsigned int i ) const;

		/**
		 * Sum buffer entries in range specified by slot and eindex
		 */
		double sumBuf( SyncId slot, unsigned int i ) const;

		/**
		 * return product of v with all buffer entries in range specified 
		 * by slot and eindex. If none, return v.
		 */
		double prdBuf( SyncId slot, unsigned int i, double v ) const;

		/**
		 * Get the buffer pointer specified by slot and eindex.
		 * We assume that the message already defines exactly how many
		 * bytes are to go, so we don't need to get the range of buffer
		 * locations available to this message slot.
		 */
		double* getBufPtr( SyncId slot, unsigned int i );

		/**
		 * Returns the data on the specified index for the Element
		 */
		Data* data( unsigned int index );

		unsigned int numData() const {
			return numData_;
		}

		/** 
		 * This function pushes a function request onto a queue.
		 * In multithread mode it figures out which queue to use.
		 */
		void addToQ( const Qinfo& qi, const char* arg );

		/**
		 * We'll try these out as alternate Send functions, given that
		 * the buffer is local.
		 */
		void ssend1( SyncId slot, unsigned int i, double v );
		void ssend2( SyncId slot, unsigned int i, double v1, double v2 );

		const Conn& conn( ConnId c ) const;

		/** 
		 * Pushes the Msg m onto the list, and returns the index to look
		 * it up as the MsgId.
		 */
		MsgId addMsg( Msg* m );

	private:
		const Msg* getMsg( MsgId mid ) const;
		/**
		 * These are the objects managed by the Element
		 * Option 1: Just have a big linear array of objects. 
		 * 		Lookup using size info. Substantial savings possible.
		 * Option 2: Allocate the objects elsewhere, just ptrs here.
		 * 		Easy to get started with.
		 */
		Data* d_;
		unsigned int numData_;
		unsigned int dataSize_;
		// vector< Data* > d_;

		/**
		 * This is the data buffer used for outgoing sync messages from
		 * objects on this Element.
		 * At creation time the objects know exactly how much buffer space
		 * they need, from the Finfos.
		 * Align as doubles because most fast data transfer is doubles.
		 */
		double* sendBuf_;
		

		/**
		 * This holds the pointers to the data buffers.
		 * Align as doubles because most fast data transfer is doubles.
		 */
		vector< double* > procBuf_;

		/**
		 * This looks up entries in the procBuf, based on msg slot and
		 * Element index.
		 */
		vector< unsigned int > procBufRange_; // Size of this is static.

		/**
		 * Number of outgoing sync msg slots. Used to work out indexing into
		 * send buffer.
		 */
		unsigned int numSendSlots_;

		/**
		 * Number of incoming sync msg slots. Used to work out indexing into
		 * ProcBufRange.
		 */
		unsigned int numRecvSlots_;

		/**
		 * This is the buffer for incoming async function requests to this 
		 * Element. Entries are organized as FuncId, MsgId, data.
		 * The FuncId has implicit knowledge of data size.
		 */
		vector< char > q_; // incoming request queue.

		/**
		 * Class information
		 */
		const Cinfo* cinfo_;

		/**
		 * Message vector. This is the low-level messaging information.
		 */
		vector< Msg* > m_;

		/**
		 * Connection vector. Connections are mid-level messaging info.
		 * They group together messages to be called by a given 'send'
		 * command.
		 */
		vector< Conn > c_;
};
