
/**
 * Have a single Element class, that uses an IndexRange. There can
 * be multiple such objects representing a complete array. So the Id
 * must know how to access them all. Or they each refer to each other.
 * Or the user only sees the master (but may be many if distrib over nodes)
 */
class Element
{
	friend int main();
	public:
		Element( Data *d )
			: d_( d )
			{;}
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
		void clearQ();
		 */

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
