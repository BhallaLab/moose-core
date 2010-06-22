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
		/**
		 * This constructor is used when making zombies. We want to have a
		 * temporary Element for field access but nothing else, and it
		 * should not mess with messages or Ids.
		 */
		Element( const Cinfo* c, DataHandler* d );

		/// Regular constructor
		Element( Id id, const Cinfo* c, const string& name,
			const vector< unsigned int >& dimensions, 
			bool isGlobal = 0 );

		/// Regular constructor
		Element( Id id, const Cinfo* c, const string& name,
			DataHandler* dataHandler );

		/**
		 * This constructor copies over the original n times.
		 */
		Element( Id id, const Element* orig, unsigned int n );

		/**
		 * Destructor
		 */
		~Element();

		/**
		 * Returns name of Element
		 */
		const string& getName() const;
		
		/**
		 * Changes name of Element
		 */
		void setName( const string& val );

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
		 * Returns the DataHandler, which actually manages the data.
		 */
		DataHandler* dataHandler() const;

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
			const ProcInfo *p, const char* arg ) const;

		/**
		 * Asynchronous send command, going to specific target Element/Data.
		 * Adds Qinfo and data onto msg specified
		 * by bindIndex, and queue specified in the ProcInfo.
		 */
		void tsend( Qinfo& q, BindIndex bindIndex, const ProcInfo *p, 
			const char* arg, const FullId& target ) const;

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

		/**
		 * Destroys all Elements in tree, being efficient about not
		 * trying to traverse through clearing messages to doomed Elements.
		 * Assumes tree includes all child elements.
		 * Typically the Neutral::destroy function builds up this tree
		 * and then calls this function.
		 */
		static void destroyElementTree( const vector< Id >& tree );

		/**
		 * Returns the Id on this Elm
		 */
		Id id() const;

		/**
		 * Returns the Msg that calls the specified Fid, on current Element.
		 * Returns 0 on failure.
		 */
		 MsgId findCaller( FuncId fid ) const;

		/**
		 * Returns the binding index of the specified entry.
		 * Returns ~0 on failure.
		 */
		 unsigned int findBinding( MsgFuncBinding b ) const;

		/**
		 * zombieSwap: replaces the Cinfo and DataHandler of the zombie.
		 * Deletes old DataHandler first.
		 */
		void zombieSwap( const Cinfo* newCinfo, DataHandler* newDataHandler     );

	private:
		string name_;

		Id id_; // The current way of accessing Id.

		/**
		 * This object stores and manages the actual data for the Element.
		 */
		DataHandler* dataHandler_;

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
