
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
	friend void testSyncArray( unsigned int );
	public:
		/**
		 * Constructor
		 * It would be nice to have shared data: e.g., thresh and tau for
		 * IntFire. Common but not static.
		 * Also think about parent-child hierarchy.
		 */
		Element( vector< Data * >& d, 
			unsigned int numSendSlots_,
			unsigned int numRecvSlots_
		);

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
		 * Reinitialize things.
		 */
		void reinit();

		/**
		 * Clear sporadic message queue, events that come randomly.
		 */
		void clearQ( const char* buf );

		/**
		 * Utility function for calling function with specified FuncId.
		 * May need to be optimized out.
		 */
		unsigned int execFunc( FuncId f, const char* buf );

		/**
		 * Return a single buffer entry specified by slot and eindex
		 */
		double oneBuf( Slot slot, unsigned int i );

		/**
		 * Sum buffer entries in range specified by slot and eindex
		 */
		double sumBuf( Slot slot, unsigned int i );

		/**
		 * return product of v with all buffer entries in range specified 
		 * by slot and eindex. If none, return v.
		 */
		double prdBuf( Slot slot, unsigned int i, double v );

		/**
		 * Get the buffer pointer specified by slot and eindex.
		 * We assume that the message already defines exactly how many
		 * bytes are to go, so we don't need to get the range of buffer
		 * locations available to this message slot.
		 */
		double* getBufPtr( Slot slot, unsigned int i );

		/**
		 * Returns the data on the specified index for the Element
		 */
		Data* data( unsigned int index );

		/** 
		 * This function pushes a synaptic event onto a queue.
		 * It is meant to be thread-safe: multiple threads can call it,
		 * but only one thread is permitted to remove the queue entries.
		 */
		void addSpike( unsigned int elementIndex, 
			unsigned int synId, double time );

		const vector< Msg* >& msg( Slot slot ) const;

		/**
		 * We'll try these out as alternate Send functions, given that
		 * the buffer is local.
		 */
		void send1( Slot slot, unsigned int i, double v );
		void send2( Slot slot, unsigned int i, double v1, double v2 );

	private:
		/**
		 * These are the objects managed by the Element
		 * Option 1: Just have a big linear array of objects. 
		 * 		Lookup using size info. Substantial savings possible.
		 * Option 2: Allocate the objects elsewhere, just ptrs here.
		 * 		Easy to get started with. Using this for now.
		 */
		vector< Data* > d_;

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
		/*
		vector< MsgSrc > m_;
		vector< MsgBuf > b_;
		vector< char >generalQ_; // This grows as needed.
		vector< char >processQ_; // This is set by # of incoming proc msgs
		*/

		/**
		 * This points to the Queue for incoming synaptic events,
		 * maintained on the Msg.
		SynQ* synQ;
		 */

		Finfo** finfo_;

		// srcMsg[ slot ] is the list of msgs emanating from 
		// the specified slot. Note that each 
		/**
		 * These are messages emanating from this Element.
		 * Each Msg is an array message: the specific element indices
		 * are handled within the Msg.
		 * Indexing:
		 * msg_[ slot ][ msgNo ]
		 * where slot determines message identity
		 * and msgNo counts distinct sets of targets within a slot.
		 */
		vector< vector< Msg* > > msg_;

		/**
		 *
		 * map< key, T >
		 *
		 * map< unsigned int slot, unsigned int msgVecIndex >
		 */
//		 map< unsigned int, unsigned int > msgMap_;

		/**
		 * Another option is a sparse matrix. Works a bit better for
		 * very sparse connectivity.
		 */

		/**
		 * Number of outgoing msg slots. Used to work out indexing into
		 * send buffer.
		 */
		unsigned int numSendSlots_;

		/**
		 * Number of incoming msg slots. Used to work out indexing into
		 * ProcBufRange.
		 */
		unsigned int numRecvSlots_;

};

/*
class BufferInfo
{
	public:
		const char* begin;
		const char* end;
};

class Eref
{
	public:
		BufferInfo processBuffer( Slot slot );
		BufferInfo asyncBuffer(); // No slots, go through all pending items
};
*/
