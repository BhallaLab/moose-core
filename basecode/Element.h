/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2009 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

class SrcFinfo;
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
	friend void Id::initIds();
	public:
		/**
		 * This constructor is used when making zombies. It is used when
		 * we want to have a
		 * temporary Element for field access but nothing else, and it
		 * should not mess with messages.
		 */
		Element( Id id, const Cinfo* c, DataHandler* d );

		/**
		 * This is the main constructor, used by Shell::innerCreate
		 * which makes most Elements. Also used to create base
		 * Elements to init the simulator in main.cpp.
		 * Id is the Id of the new Element
		 * Cinfo is the class
		 * name is its name
		 * dimensions specify the cumulative dimensions all the way up
		 * to the current Element. So if you want to create an
		 * Element 'synapse' as follows:
		 * /bulb[0..1]/glom[0..1999]/mit[0..24]/dend[0..99]/synapse
		 * the dimensions would be {2,2000,25,100,1}
		 * Dimensions also allows you specify when a given dimension should
		 * be ragged, that is, contain differently sized sub arrays. A
		 * ragged array dimension has a negative sign. For example,
		 * supposing we permitted anywhere up to 50 mitral cells per glom,
		 * we would have dimensions[] = {2,2000,-50,100,1}
		 * The isGlobal flag specifies whether the created objects should
		 * be replicated on all nodes, or partitioned without replication. 
		 */
		Element( Id id, const Cinfo* c, const string& name,
			const vector< DimInfo >& dimensions, unsigned short pathDepth,
			bool isGlobal = 0 );

		/**
		 * This constructor is used when making FieldElements.
		 * It allows the user to explicitly specify the DataHandler
		 * to be used.
		 */
		Element( Id id, const Cinfo* c, const string& name,
			DataHandler* dataHandler );

		/**
		 * This constructor copies over the original n times. It is
		 * used for doing all copies, in Shell::innerCopyElements.
		 */
		Element( Id id, const Element* orig, unsigned int n, 
			unsigned short newParentDepth,
			unsigned short copyRootDepth, bool toGlobal);

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
		 * Returns group in which Element belongs
		 */
		unsigned int getGroup() const;
		
		/**
		 * Changes group of Element
		 */
		void setGroup( unsigned int val );

		/**
		 * Examine process queue, data that is expected every timestep.
		 * This function is done for all the local data entries in order.
		 */
		void process( const ProcInfo* p, FuncId fid );


		/**
		 * Returns the DataHandler, which actually manages the data.
		 */
		DataHandler* dataHandler() const;

		/**
		 * Resizes the data at the specified depth on the Element tree,
		 * Returns true on success.
		 * Adds a dimension if the original was zero and size > 1.
		 * When resizing it uses the current data and puts it treadmill-
		 * fashion into the expanded dimension. 
		 */
		bool resize( unsigned short pathDepth, unsigned int size );

		/**
		 * Synchronizes the maxFieldEntries with the number of actual fields
		 * in the parent. 
		 * Also will need to synchronize across nodes, in due course.
		 */
		void syncFieldDim() const;

		/**
		 * Asynchronous send command. Adds Qinfo and data onto msg specified
		 * by bindIndex, and queue specified in the ProcInfo.
		void asend( Qinfo& q, BindIndex bindIndex, 
			const ProcInfo *p, const char* arg ) const;
		 */

		/**
 		 * Executes a queue entry from the buffer.
 		 */
		void exec( const Qinfo* qi, const double* arg ) const;

		/**
		 * Asynchronous send command, going to specific target Element/Data.
		 * Adds Qinfo and data onto msg specified
		 * by bindIndex, and queue specified in the ProcInfo.
		void tsend( Qinfo& q, BindIndex bindIndex, const ProcInfo *p, 
			const char* arg, const ObjId& target ) const;
		 */

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
		 * Returns true if there are one or more Msgs on the specified
		 * BindIndex
		 */
		bool hasMsgs( BindIndex b ) const;

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


	/////////////////////////////////////////////////////////////////////
	// Utility functions for message traversal
	/////////////////////////////////////////////////////////////////////
		/**
		 * Returns the binding index of the specified entry.
		 * Returns ~0 on failure.
		 */
		 unsigned int findBinding( MsgFuncBinding b ) const;

		 /**
		  * Returns all incoming Msgs.
		  */
		 const vector< MsgId >& msgIn() const;

		/**
		 * Returns the first Msg that calls the specified Fid, 
		 * on current Element.
		 * Returns 0 on failure.
		 */
		 MsgId findCaller( FuncId fid ) const;

		/** 
		 * More general function. Fills up vector of MsgIds that call the
		 * specified Fid on current Element. Returns # found
		 */
		unsigned int getInputMsgs( vector< MsgId >& caller, FuncId fid)
		 	const;

		/**
		 * Fills in vector of Ids connected to this Finfo on
		 * this Element. Returns # found
		 */
		unsigned int getNeighbours( vector< Id >& ret, const Finfo* finfo )
			const;

		/**
		 * Fills in vector, each entry of which identifies the src and 
		 * dest fields respectively. 
		 * Src field is local and identified by BindIndex
		 * Dest field is a FuncId on the remote Element.
		 */
		unsigned int getFieldsOfOutgoingMsg( 
			MsgId mid, vector< pair< BindIndex, FuncId > >& ret ) const;

		/**
		 * zombieSwap: replaces the Cinfo and DataHandler of the zombie.
		 * Deletes old DataHandler first.
		 */
		void zombieSwap( const Cinfo* newCinfo, DataHandler* newDataHandler     );

	private:
		/**
		 * Fills in vector of Ids receiving messages from this SrcFinfo. 
		 * Returns # found
		 */
		unsigned int getOutputs( vector< Id >& ret, const SrcFinfo* finfo )
			const;

		/**
		 * Fills in vector of Ids sending messeges to this DestFinfo on
		 * this Element. Returns # found
		 */
		unsigned int getInputs( vector< Id >& ret, const DestFinfo* finfo )
			const;

		string name_;

		Id id_; /// Stores the unique identifier for Element.

		/**
		 * This object stores and manages the actual data for the Element.
		 */
		DataHandler* dataHandler_;

		/**
		 * Class information
		 */
		const Cinfo* cinfo_;

		/**
		 * Identifies which closely-connected group the Element is in.
		 * Each Group is assumed to have dense message traffic internally,
		 * and uses MPI_Allgather to exchange data.
		 * Not yet in use.
		 */
		 unsigned int group_;

		/**
		 * Message vector. This is the low-level messaging information.
		 * Contains info about incoming as well as outgoing Msgs.
		 */
		vector< MsgId > m_;

		/**
		 * Binds an outgoing message to its function.
		 * Each index (BindIndex) gives a vector of MsgFuncBindings,
		 * which are just pairs of MsgId, FuncId.
		 * SrcFinfo keeps track of the BindIndex to look things up.
		 * Note that a single BindIndex may refer to multiple Msg/Func
		 * pairs. This means that a single MsgSrc may dispatch data 
		 * through multiple msgs using a single 'send' call.
		 */
		vector< vector < MsgFuncBinding > > msgBinding_;
};
