
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
	public:
		/**
		 * Constructor
		 */
		Element( Data *d );

		/**
		 * Examine process queue, data that is expected every timestep.
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
		void pushQ( unsigned int elementIndex, 
			unsigned int synId, double time );

		const vector< Msg* >& msg( Slot slot ) const;

	private:
		Data* d_;

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
		 * Another option is a sparse matrix. Works a bit better for
		 * very sparse connectivity.
		 */

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
