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
	friend void testSparseMsg();
	public:
		enum Decomposition {
			Block,	// Successive blocks of Elements are on same node
			Sequential // Entries are put sequentially on successive nodes.
		};
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
		/*
		Element( const Cinfo* c, 
			char* d, unsigned int numData, unsigned int dataSize,
			unsigned int numBindIndex, Element::Decomposition decomp );
			*/

		Element( Id id, const Cinfo* c, const string& name,
			unsigned int numData, Element::Decomposition decomp =
				Element::Block );

		// What is the point of this constructor?
		Element( const Cinfo* c, const Element* other );

		/**
		 * Destructor
		 */
		virtual ~Element();

		const string& name() const;

		/**
		 * Here we build the array on the fly.
		Element( const Data *prototype, unsigned int numEntries );
		 */

		/**
		 * Examine process queue, data that is expected every timestep.
		 * This function is done for all the local data entries in order.
		 */
		virtual void process( const ProcInfo* p );

		/**
		 * Clear sporadic message queue, events that come randomly.
		void clearQ();
		 */

		/**
		 * execFunc executes the function defined in the buffer, and
		 * returns the next position of the buffer. Returns 0 at the
		 * end.
		const char* execFunc( const char* buf );
		 */

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
		 * Here we may lose the nice orthogonality of the Data and of the
		 * Elements. If we have Elements that handle arrays of fields,
		 * then they have to know about the data structures holding those
		 * fields. For example, an Element that deals with Synapses must
		 * know how to get a Synapse from an IntFire.
		 */
		virtual char* data( DataId index );

		/**
		 * Returns the data at one level up of indexing. So, for a 2-D
		 * array, would return the start entry of the rows rather than
		 * the individual entries. For a synapse on an IntFire, would
		 * return the appropriate IntFire, rather than the synapse.
		 */
		virtual char* data1( DataId index );

		/**
		 * Returns the number of data entries
		 */
		virtual unsigned int numData() const;

		/**
		 * Returns the number of data entries at index 1.
		 * For regular Elements this is identical to numData
		 * For Elements whose entries are array fields, this is
		 * the number of parent objects.
		 */
		virtual unsigned int numData1() const;

		/**
		 * Returns the number of data entries at index 2, if present.
		 * For regular Elements this is always 1.
		 * For Elements whose entries are array fields, this is the number
		 * of fields on that data entry.
		 */
		 virtual unsigned int numData2( unsigned int index1 ) const;

		/**
		 * Returns the number of dimensions of the data.
		 */
		virtual unsigned int numDimensions() const;

		/**
		 * Assigns the sizes of all array field entries at once.
		 * This is ignored for regular Elements.
		 * In a FieldElement we can assign different array sizes for 
		 * each entry in the Element.
		 * Note that a single Element may have more than one array field.
		 * However, each FieldElement instance will refer to just one of
		 * these array fields, so there is no ambiguity.
		 */
		virtual void setArraySizes( const vector< unsigned int >& sizes );

		/**
		 * Looks up the sizes of all array field entries at once. Returns
		 * all ones for regular Elements. 
		 * Note that a single Element may have more than one array field.
		 * However, each FieldElement instance will refer to just one of
		 * these array fields, so there is no ambiguity.
		 */
		virtual void getArraySizes( vector< unsigned int >& sizes ) const;

		/**
		 * We'll try these out as alternate Send functions, given that
		 * the buffer is local.
		 */
		void ssend1( SyncId slot, unsigned int i, double v );
		void ssend2( SyncId slot, unsigned int i, double v1, double v2 );

		/**
		 * Asynchronous send command. Adds Qinfo and data onto msg specified
		 * by bindIndex, and queue specified in the ProcInfo.
		 */
		void asend( Qinfo& q, BindIndex bindIndex, 
			const ProcInfo *p, const char* arg );

		/**
		 * Asynchronous send command, going to specific target Element/Data.
		 * Adds Qinfo and data onto msg specified
		 * by bindIndex, and queue specified in the ProcInfo.
		 */
		void tsend( Qinfo& q, BindIndex bindIndex, 
			const ProcInfo *p, const char* arg, const FullId& target );

		/** 
		 * Pushes the Msg mid onto the list.
		 * The position on the list does not matter.
		 * 
		 */
		void addMsg( MsgId mid );

		/**
		 * Removes the specified msg from the list.
		 */
		void dropMsg( MsgId mid );
		
		/**
		 * Clears out all Msgs on specified BindIndex. Used in Shell::set
		 */
		void clearBinding( BindIndex b );

		/**
		 * Pushes back the specified Msg and Func pair into the properly
		 * indexed place on the msgBinding_ vector.
		 */
		void addMsgAndFunc( MsgId mid, FuncId fid, BindIndex bindIndex );

		/**
		 * gets the Msg/Func binding information for specified bindIndex.
		 * This is a vector.
		 * Returns 0 on failure.
		 */
		const vector< MsgFuncBinding >* getMsgAndFunc( BindIndex b ) const;

		/**
		 * Utility function for printing out all fields and their values
		 */
		void showFields() const;

		/**
		 * Utility function for traversing and displaying all messages
		 */
		void showMsg() const;

		/**
		 * Gets the class information for this Element
		 */
		const Cinfo* cinfo() const;

		// const Msg* getMsg( MsgId mid ) const;
		
		Decomposition decomposition() const;

	protected:

		/**
		 * These are the objects managed by the Element
		 * Option 1: Just have a big linear array of objects. 
		 * 		Lookup using size info. Substantial savings possible.
		 * 		Easier to manage memory.
		 * Option 2: Allocate the objects elsewhere, just ptrs here.
		 * 		Easy to get started with.
		 */
		char* d_;
		unsigned int numData_;	// Number of data entries
		unsigned int dataSize_;	// Size of each data entry
		unsigned int dataStart_;	// Starting index of data, used in MPI.

		// Enum. Specifies how to subdivide indices among nodes.
		Decomposition decomposition_; 
	private:
		string name_;

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
		 * Element. Entries are organized as Qinfo, data.
		 * Qinfo explicitly stores the size of the data.
		vector< char > q_; // incoming request queue.
		 */

		/**
		 * Class information
		 */
		const Cinfo* cinfo_;

		/**
		 * Message vector. This is the low-level messaging information.
		 * Contains info about incoming msgs? But this lacks binding info.
		 */
		vector< MsgId > m_;

		/**
		 * Binds an outgoing message to its function.
		 * SrcFinfo keeps track of the BindIndex to look things up.
		 * Note that a single BindIndex may refer to multiple Msg/Func
		 * pairs.
		 */
		vector< vector < MsgFuncBinding > > msgBinding_;

		/**
		 * High level messaging info, containing a declarative record of the
		 * specification of the message including connectivity and
		 * functionality. These are indices into a global vector.
		 */
//		vector< MsgSpecId >
};
