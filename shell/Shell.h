/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/
#ifndef _SHELL_H
#define _SHELL_H
#include "../basecode/Fid.h"
// forward declaration
class SimDump;

typedef struct {
	unsigned int numPending;
	void* data;
} offNodeData;

class Shell
{
#ifdef DO_UNIT_TESTS
	friend void testShell();
	friend void testOffNodeQueue();
#ifdef USE_MPI
	friend void testShellSetupAsyncParMsg();
	friend void testBidirectionalParMsg();
	friend void testParGet( Id tnId, vector< Id >& testIds );
#endif // USE_MPI
#endif // DO_UNIT_TESTS
	public:
        Shell();
	
////////////////////////////////////////////////////////////////////
// Some utility functions for path management.
// These need to be here in Shell for two reasons. First, the shell
// holds the current working element, cwe. Second, many functions are
// multinode and the Shell has to handle queries across nodes.
////////////////////////////////////////////////////////////////////
		/** 
		 * Returns a string with the fully expanded unix-like path of
		 * the specified Id. Now works in parallel, except for cases
		 * where node boundaries are crossed above the root object.
		 */
		static string eid2path( Id eid );
		
		/**
		 * This is the local node version of eid2path
		 */
		static string localEid2Path( Id eid );
		/**
		 * For parallel operation, we need to handle requests for the path
		 * from different nodes. 
		 */
		static void handlePathRequest(
				const Conn* c, 
				Nid eid,
				unsigned int requestId );
		/**
		 * For parallel operation, we need to return the path when requested
		 */
		static void handlePathReturn(
				const Conn* c, 
				string path,
				unsigned int requestId );
		
		/**
		 * Returns an Id defined by the specified path. Works in parallel,
		 * but not for cases where node boundaries are crossed above the
		 * root object.
		 * If the 'isLocal' flag is set does NOT try to go off-node.
		 */
		static Id path2eid(
				const string& path, 
				const string& separator,
				bool isLocal );
		
		/**
		 * Inner function for extracting Id.
		 */
		Id innerPath2eid(
				const string& path, 
				const string& separator,
				bool isLocal ) const;
		
		/**
		 * Deeper inner function: checks on local node then lauches out
		 * on other nodes.
		 */
		static Id traversePath(
				Id start,
				vector< string >& );
		
		/**
		 * Innermost function that does the actual Id extraction work,
		 * but only on a local node.
		 */
		static Id localTraversePath(
				Id start,
				vector< string >& );
		
		/**
		* Returns Id for off-node object specified by path.
 		* Sets off request for children on all remote nodes, starting at 
 		* either root or shellId. Polls till all nodes return. Note that
		* this must be thread-safe, because during the polling there
		* may be nested calls.
 		*/
		static Id parallelTraversePath(
				Id start,
				vector< string >& names );
		
		/**
		 * Handles request for path traversal on remote node.
		 */
		static void handleParTraversePathRequest(
				const Conn* c,
				Id start,
				vector< string > names,
				unsigned int requestId );
		
		/**
		 * Handles return value from path traversal on remote node.
		 */
		static void handleParTraversePathReturn(
				const Conn* c,
				Nid found,
				unsigned int requestId );
		
		/**
		 * Returns parent Id.
		 */
		static Id parent( Id eid );
		
		/**
		 * Returns string upto but not including last separator
		 * Returns empty string if no separator.
		 */
		static string head(
				const string& path,
				const string& separator );
		
		/**
		 * Returns string after last separator.
		 * Returns original string if no separator
		 */
		static string tail(
				const string& path,
				const string& separator );
		
		/**
		 * Converts path from relative, recent or other forms to 
		 * a canonical absolute path that the wildcards can handle
		 */
		void digestPath( string& path );
		
//////////////////////////////////////////////////////////////////////
// Special low-level operations that Shell handles using raw
// serialized strings from PostMaster.
//////////////////////////////////////////////////////////////////////
		/**
		 * Deprecated
		 */
		static void rawAddFunc( const Conn* c, string s );
		/**
		 * Deprecated
		 */
		static void rawCopyFunc( const Conn* c, string s );
		/**
		 * Deprecated
		 */
		static void rawTestFunc( const Conn* c, string s );
//////////////////////////////////////////////////////////////////////
// Infinite loop called on slave nodes to monitor commands from master.
//////////////////////////////////////////////////////////////////////
		static void pollFunc( const Conn* c );
		
//////////////////////////////////////////////////////////
// Parallel information
//////////////////////////////////////////////////////////
		/// Reports if we are running in serial / parallel mode
		static bool isSerial( );
		
		/// returns node # of shell.
		static unsigned int myNode(); 
		
		/// Returns # of nodes in simulation
		static unsigned int numNodes(); 
		
		/// Used only in setup phase. Assigns node info.
		static void setNodes(
				unsigned int myNode,
				unsigned int numNodes );
		
////////////////////////////////////////////////////////////////////
// Id management
////////////////////////////////////////////////////////////////////
		static unsigned int newIdBlock( unsigned int size );
		
		static void handleRequestNewIdBlock(
				const Conn* c,
				unsigned int size,
				unsigned int node,
				unsigned int requestId );
		
		static void handleReturnNewIdBlock(
				const Conn* c,
				unsigned int value,
				unsigned int requestId );
		
////////////////////////////////////////////////////////////////////
// Local functions for implementing basic GENESIS/MOOSE command set.
////////////////////////////////////////////////////////////////////
		static void setCwe( const Conn*, Id id );
		static Id getCwe( Eref e );
		static void trigCwe( const Conn* );
		static void pushe( const Conn*, Id id );
		static void pope( const Conn* );
		
		static int getMyNode( Eref e );
		static int getNumNodes( Eref e );
		
		static void trigLe( const Conn*, Id parent );
		static void handleRequestLe(
				const Conn* c, 
				Nid parent,
				unsigned int requestId );
		/**
 		* Undefined effects if more than one node has a matching target.
 		*/
		static void handleReturnLe(
				const Conn* c,
				vector< Nid > found,
				unsigned int requestId );
		
		Element* create(
				const string& type,
				const string& name,
				Id parent,
				Id id );
		
		Element* createArray(
				const string& type,
				const string& name,
				Id parent,
				Id id,
				int n );
		
		void destroy( Id victim );
		
		/**
		 * This function creates an object, generating its
		 * own Id. Node argument tells it which node to put it on,
		 * but if this is Id::UnknownNode then the placement is left to the 
		 * system balancing algorithm.
		 */
		static void staticCreate(
				const Conn*,
				string type,
				string name,
				unsigned int node,
				Id parent );
		
		/*
		static void staticCreateArray1(
				const Conn*,
				string type,
				string name,
				Id parent,
				vector <double> parameter );
		*/
		
		static void staticCreateArray(
				const Conn*,
				string type,
				string name,
				Id parent,
				vector <double> parameter );
		
		static Element* createGlobal(
				const string& type,
				const string& name,
				Id parent,
				Id id );
		
		static void planarconnect(
				const Conn* c,
				string source,
				string dest,
				double probability);
		
		static void planardelay(
				const Conn* c,
				string source,
				string destination,
				vector< double > parameter);
		
		static void planarweight(
				const Conn* c,
				string source,
				string destination,
				vector< double > parameter);
		
		static void getSynCount2(
				const Conn* c,
				Id dest);
		
		/**
		 * Delete an object. Works in parallel.
		 */
		static void staticDestroy( const Conn*, Id victim );
		
		////////////////////////////////////////////////////////////////
		// Group of functions for field access
		////////////////////////////////////////////////////////////////
		/**
		 * Gets a field value. Works in parallel. Blocks.
		 * Triggers a return
		 * function on the remote Shell, and manages a return value queue
		 * so that it can run in a thread-safe manner. Sets up a
		 * non-preemptive busy loop to poll the postmaster, so it should
		 * not result in a race condition. Ideally would do this using
		 * a separate thread.
		 */
		static void getField(
				const Conn* c,
				Id id,
				string field );
		
		/**
		 * Invoked by addField on appropriate nodes, and does the actual adding
		 * of the ExtFieldFinfo.
		 */
		static void localAddField(
				const Conn* c,
				Id id,
				string fieldname );
		
		/**
		 * Calls localAddField on target nodes, to add new fields on objects.
		 */
		static void addField(
				const Conn* c,
				Id id,
				string fieldname );
		
		/**
		 * Assign a field value to the id on the local node.
		 */
		static void localSetField(
				const Conn* c, 
				Id id,
				string field,
				string value );
		
		/**
		 * Assign a field value to an id on any node, local or remote.
		 * Does not block: just issues the request, and carries on.
		 * The remote node executes a localSetField to handle the request.
		 */
		static void setField(
				const Conn* c, 
				Id id,
				string field,
				string value );
		/**
		 * Assigns a field value to a vector of ids, on any node.
		 * Does not block: just issues the request, and carries on.
		 * Currently does this in a simple, non-optimized way by calling
		 * setField lots of times.
		 */
		static void setVecField(
				const Conn* c, 
				vector< Id > elist,
				string field,
				string value );
		
		////////////////////////////////////////////////////////////////
		// Group of functions for scheduling
		////////////////////////////////////////////////////////////////
		static void setClock(
				const Conn* c,
				int clockNo,
				double dt,
				int stage );
		
		static void useClock(
				const Conn* c,
				string tickName,
				string path,
				string function );
		
		static void localUseClock(
				const Conn* c,
				string tickName,
				string path,
				string function );
		
		static void innerUseClock( 
				Id tickId,
				vector< Id >& path,
				string function );
		
		////////////////////////////////////////////////////////////////
		// Group of functions for wildcards. Also works in parallel
		////////////////////////////////////////////////////////////////
		/**
		 * Searches all nodes for objects matching the wildcard path.
		 * Sends back the list on the a return message.
		 */
		static void getWildcardList(
				const Conn* c,
				string path,
				bool ordered );
		
		/**
		 * Searches a single node for objects matching the wildcard path
		 * Handles off-node cases too.
		 */
		static void innerGetWildcardList(
				const Conn* c,
				string path,
				bool ordered,
				vector< Id >& list );
		
		/**
		 * Does the actual work of getting the wildcard list, on the
		 * local node only.
		 */
		static void localGetWildcardList(
				const Conn* c,
				string path,
				bool ordered,
				vector< Id >& list );
		
		/**
		 * Deals with off-node requests for a wildcard list
		 */
		static void handleParWildcardList(
				const Conn* c,
				string path,
				bool ordered,
				unsigned int requestId );
		
		////////////////////////////////////////////////////////////////
		// Group of functions for messaging. Also works in parallel
		////////////////////////////////////////////////////////////////
		
		/**
		 * This is called from the same node that the src is on, to send a 
		 * message to a dest on a remote node. 
		 * Note that an Id does not carry node info within itself. So we
		 * use an Nid for the dest, as we need to retain node info
		 */
		static void addParallelSrc(
				const Conn* c,
				Nid src,
				string srcField,
				Nid dest,
				string destField );
		
		/**
		 * Operates on the node of the destination object to complete the
		 * message setup across nodes. This is the counterpart of 
		 * addParallelSrc. Creates a proxy object if needed on the target
		 * node.
		 */
		static void addParallelDest(
				const Conn* c,
				Nid src,
				unsigned int srcSize,
				string srcTypeStr, 
				Nid dest,
				string destField );
		
		/**
		 * addMessage creates a message between two Ids, which could
		 * also represent array elements.
		 */
		static void addMessage(
				const Conn* c,
				vector< Id >src,
				string srcField,
				vector< Id >dest,
				string destField );
		
		static bool addSingleMessage(
				const Conn* c,
				Id src,
				string srcField,
				Id dest,
				string destField );
		/** 
		 * This one does the actual work for adding messages on the
		 * local node
		 */
		static bool innerAddLocal(
				Id src,
				string srcField,
				Id dest,
				string destField );
		
		/**
		 * Wrapper function for innerAddLocal so that messages can use it.
		 */
		static bool addLocal(
				const Conn* c,
				Id src, string srcField,
				Id dest, string destField );
		
		/**
		 * This is a more general form of the addMessage command,
		 * as it lets the user specify the connection type. Currently
		 * not available in the GENESIS parser.
		 */
		static void addEdge(
				const Conn* c,
				Fid src,
				Fid dest,
				int connType );
		
		/**
		 * deleteMessage gets rid of the message identified by the Fid and
		 * the integer lookup for it. Not parallel.
		 */
		static void deleteMessage(
				const Conn* c,
				Fid src,
				int msg );
		
		/**
		 * deleteEdge gets rid of messages specified as edges, that is,
		 * using the same src/field and dest/field info that was used
		 * to create the message. Not parallel.
		 */
		static void deleteMessageByDest(
				const Conn* c,
				Id src,
				string srcField,
				Id dest,
				string destField );
		
		/**
		 * Delete a message, in a more general manner. Not parallel.
		 */
		static void deleteEdge(
				const Conn* c,
				Fid src,
				Fid dest );
		
		/**
 		* listMessages builds a list of messages associated with the 
 		* specified element on the named field, and sends it back to
 		* the calling parser. It extracts the
 		* target element from the connections, and puts this into a
 		* vector of unsigned ints.
 		*/
		static void listMessages(
				const Conn* c,
				Id id,
				string field,
				bool isIncoming );
				
		/**
		 * Does the actual work to build message list, on a local node.
		 */
		static void innerListMessages(
				const Conn* c,
				Id id,
				string field,
				bool isIncoming,
				vector< Id >& ret,
				string& remoteFields );
		
		//////////////////////////////////////////////////////////
		// Functions for handling moves and copies. Not yet parallelized
		//////////////////////////////////////////////////////////
		/**
		 * There are two versions of copy, selected by #ifdef USE_MPI.
		 * The single node version just makes a local copy.
		 * The multinode version is still a bit skeletal. It doesn't 
		 * handle any cases of copying between nodes, including cases
		 * where the target is a global. On the other hand it is OK with
		 * copying globals to globals, and copying on remote nodes
		 * provided src and dest are on the same node. 
		 * Another current limitation is that it does not return the 
		 * new object Id in cases where the object creation is off-node.
		 */
		static void copy(
				const Conn* c,
				Id src,
				Id parent,
				string name );
		
		/**
		 * This function copies the prototype element in form of an array.
		 * It is similar to copy() only that it creates an array of copied 
		 * elements. 
		 * Implemented as two different versions, depending on whether we
		 * are running in parallel mode or not. Like copy, some limitations
		 * apply to the parallel version: Cannot copy across nodes, and
		 * cannot return new object Id if the new array is off-node.
		 */
		static void copyIntoArray(
				const Conn* c,
				Id src,
				Id parent,
				string name,
				vector< double > parameter );
		
		/**
		 * This function does the actual node-local array copy.
		 */
		static Element* localCopyIntoArray(
				const Conn* c,
				Id src,
				Id parent,
				string name,
				vector< double > parameter,
				IdGenerator idGen = IdGenerator() );
		
		/**
		 * Handles a copy on a local node. If the Id is defined, then it assigns
		 * the Id to the newly created copy. Otherwise an Id is generated
		 * locally from a block of Ids allocated to this node.
		 * 
		 * At some point this needs to be upgraded to return the created id to
		 * the master node.
		 */
		static void parCopy(
				const Conn* c,
				Nid src,
				Nid parent, 
				string name,
				IdGenerator idGen );
		
		/**
		 * This manages remote requests for copying an array.
		 * The same concerns with ensuring compatibility of Id remains,
		 * for globals. 
		 * First arg is a 3 Nid vector: src, parent, child.
		 * Later we may want to put in a callback so that we can return
		 * info on success of operation.
		 */
		static void parCopyIntoArray(
				const Conn* c,
				Nid src,
				Nid parent,
				string name,
				vector< double > parameter,
				IdGenerator idGen );
		
		static void move(
				const Conn* c,
				Id src,
				Id parent,
				string name );
		
		//////////////////////////////////////////////////////////
		// Some stuff for managing scheduling and simulation runs
		//////////////////////////////////////////////////////////
		static void resched( const Conn* c );
		
		static void reinit( const Conn* c );
		
		static void stop( const Conn* c );
		
		static void step( const Conn* c, double time );
		
		static void requestClocks( const Conn* c );
		
		static void requestCurrTime( const Conn* c );
		
		//////////////////////////////////////////////////////////
		// Major input functions.
		//////////////////////////////////////////////////////////
		static void readCell( 
					const Conn* c,
					string filename,
					string cellpath,
					vector< double > globalParms,
					unsigned int node );
		
		static void localReadCell( 
					const Conn* c,
					string filename,
					string cellname,
					vector< double > globalParms,
					Nid pa,
					IdGenerator idGen );
		
		//////////////////////////////////////////////////////////
		// Channel setup functions.
		//////////////////////////////////////////////////////////
		static void setupAlpha(
				const Conn* c,
				Id gate,
				vector< double > parms );
		
		static void setupTau(
				const Conn* c,
				Id gate,
				vector< double > parms );
		
		static void tweakAlpha(
				const Conn* c,
				Id gateId );
		
		static void tweakTau(
				const Conn* c,
				Id gateId );
		
		static void setupGate(
				const Conn* c,
				Id gate,
				vector< double > parms );
		
		//////////////////////////////////////////////////////////
		// SimDump functions
		//////////////////////////////////////////////////////////
		static void readDumpFile(
				const Conn* c,
				string filename );
		
		static void writeDumpFile(
				const Conn* c, 
				string filename,
				string path );
		
		static void simObjDump(
				const Conn* c,
				string fields );
		
		static void simUndump(
				const Conn* c,
				string args );
		
		static void openFile(
				const Conn* c,
				string filename,
				string mode );
		
		static void closeFile(
				const Conn* c,
				string filename );
		
		static void writeFile(
				const Conn* c,
				string filename,
				string text );
		
		static void flushFile(
				const Conn* c,
				string filename );
		
		static void readFile(
				const Conn* c,
				string filename,
				bool linemode );
		
		static void listFiles(
				const Conn* c );
		
		static void loadtab(
				const Conn* c,
				string data );
		
		void innerLoadTab(
				const string& data );
		
		//////////////////////////////////////////////////////////
		// Table special functions
		//////////////////////////////////////////////////////////
		static void tabop(
				const Conn* c,
				Id tab,
				char op,
				double min, 
				double max );
		
		static void file2tab(
				const Conn* c, 
				Id id,
				string filename,
				unsigned int skiplines );
		
		static void localFile2tab(
				const Conn* c, 
				Nid nid,
				string filename,
				unsigned int skiplines );
		
		//////////////////////////////////////////////////////////
		// sbml functions
		//////////////////////////////////////////////////////////
		static void readSbml(
				const Conn* c,
				string filename,
				string modelpath,
				int childnode );
		
		static void writeSbml(
				const Conn* c,
				string filename,
				string modelpath,
				int childnode );

		//////////////////////////////////////////////////////////
		// neuroml functions
		//////////////////////////////////////////////////////////
		static void readNeuroml(
				const Conn* c,
				string filename,
				string modelpath,
				int childnode );
		
		static void writeNeuroml(
				const Conn* c,
				string filename,
				string modelpath,
				int childnode );
		
		//////////////////////////////////////////////////////////
		// functions to create gates on a given channel
		//////////////////////////////////////////////////////////
		static void createGateMaster(
				const Conn* c,
				Id chan,
				string gateName );
		
		static void createGateWorker(
				const Conn* c,
				Id chan,
				string gateName,
				IdGenerator idGen );
		
		/*
		void add( const string& src, const string& dest );
		void drop( const string& src, const string& dest );
		void set( const string& field, const string& value );
		void move( const string& src, const string& dest );
		void copy( const string& src, const string& dest );
		void copyShallow( 
			const string& src, const string& dest );
		void copyHalo( const string& src, const string& dest );
		*/
		
		/*
		void listCommands( );
		void listClasses( );
		void echo( vector< string >& s, int options );
		void remoteCommand( string arglist );
		void command( int argc, const char** argv );
		*/
		
		////////////////////////////////////////////////////////////////////
		// functions for implementing inter-node field assignments.
		////////////////////////////////////////////////////////////////////
		static void parGetField(
				const Conn* c, 
				Id id,
				string field,
				unsigned int requestId );
		
		static void handleReturnGet(
				const Conn* c, 
				string value,
				unsigned int requestId );
		
		// setField is a regular command, no special stuff for it.
		
		/**
		 * Create an object on a remote node
		 */
		static void parCreateFunc(
				const Conn* c,
				string objtype,
				string objname, 
				Nid parentId,
				Nid newObjId );
		
		/**
		 * Create an array element on a remote node
		 */
		static void parCreateArrayFunc(
				const Conn* c,
				string objtype,
				string objname,
				pair< Nid, Nid > nids,
				vector< double > parameter );
		
		static void parMsgErrorFunc(
				const Conn* c,
				string errMsg,
				Id src,
				Id dest );
		
		static void parMsgOkFunc(
				const Conn* c,
				Id src,
				Id dest );
		
		/**
		 * Find how big a remote element array is. Returns 1 if it is
		 * a simpleElement.
		 */
		static unsigned int getNumDestEntries( Nid dest );
		/**
		 * Returns the Eref for the specified postmaster.
		 */
		// Eref getPost( unsigned int node ) const;
		
		/////////////////////////////////////////////////////////////////
		//	Set of commands for managing off-node requests. These
		//	act in conjunction with a couple of templated functions
		//	defined below.
		/////////////////////////////////////////////////////////////////
		
		/**
 		* The next two functions should always be called in pairs and should
 		* be called within the same function, so that local variables do not
 		* get lost. One normally uses the templated wrapper functions 
		* defined at the bottom.
 		*
 		* openOffNodeValueRequest:
 		* Inner function to handle requests for off-node operations
		* returning values.
 		* Returns a thread-safe unique id for the request, so that 
 		* we can examine the return values for ones we are interested in. 
 		* This rid is an index to a vector of ints that counts pending 
 		* returns on this id.
 		* Returns the next free Rid and initializes returnedData_ entry.
 		*
 		* The init argument is typically a local variable whose value will
 		* be read out in the succeeding extractOffNodeValue call. If these
 		* are not in the same function, then the user has to use allocated
 		* memory.
 		*/
		unsigned int openOffNodeValueRequestInner( 
				void* init,
				unsigned int numPending );
		
		/**
		 * closeOffNodeValueRequest
 		 * Polls postmaster, converts and returns data stored at rid
		 * Used as the inner body of the templated function by the
		 * same name, defined below.
 		 */
		void* closeOffNodeValueRequestInner( unsigned int rid );
		
		/**
		 * Returns the value pointer for the specified rid.
		 */
		void* getOffNodeValuePtrInner( unsigned int rid );
		/**
 		* Decrements the off node pending count.
 		*/
		void decrementOffNodePending( unsigned int rid );
		/**
 		* Zeroes out the off node pending count.
 		*/
		void zeroOffNodePending( unsigned int rid );
		
		/**
 		* Returns number of pending responses to off node request.
 		*/
		unsigned int numPendingOffNode( unsigned int rid );
		
		/**
		 * Flag: true till simulation quits. Used in the main loop.
		 */
		static bool running();
		
		/**
		 * Tells all nodes to quit, then quits.
		 */
		static void quit( const Conn* c );
		
		/**
		 * Sets running flag to 0, to get the simulator to quit
		 */
		static void innerQuit( const Conn* c );
		
		/**
		 * Used only in unit tests, when we spoof an off-node postmaster
		 * using this hack.
		 */
#ifdef DO_UNIT_TESTS
		Eref getPostForUnitTests( unsigned int node ) const;
		Element* post_;
#endif
	
	private:
		/// Current working element
		Id cwe_;
		/// Most recently created element
		Id recentElement_;
		vector< Id > workingElementStack_;
		// True if prompts etc are to be printed.
		///stores all the filehandles create by genesis style format.
		static map <string, FILE*> filehandler;
		// variable for file handling
		static vector <string> filenames;
		static vector <string> modes;
		static vector <FILE*> filehandles;
		
		int isInteractive_;
		string parser_;
		SimDump* simDump_;
		Id lastTab_; // Used for the loadtab -continue option, which 
		// contines loading numbers into the previously selected table.
		
		/// Node on which this shell operates
		static unsigned int myNode_;
		
		/// Number of nodes used by simulation.
		static unsigned int numNodes_;
		
		/// Number of requests allowed for off-node data transfer.
		static const unsigned int maxNumOffNodeRequests;
		
		// Flag for main loop of simulator. When it becomes false,
		// the simulator will exit.
		static bool running_;
		
		/**
		 * This keeps track of message requests sent off-node. 
		 * When the target node finishes its work it tells this one that
		 * things worked out. At reset time we examine this map to make
		 * sure everything has been cleared up.
		 * We can't do an immediate return because
		 * it is slow, and quite likely to give rise to race conditions.
		 */
		map< Id, Id > parMessagePending_;
		
		/////////////////////////////////////////////////////////////
		// This set of definitions is to manage return values from other
		// nodes in a thread-safe manner. Each value request generates
		// and rid, which is used to keep track of where the return value
		// should go.
		/////////////////////////////////////////////////////////////
		/// Manages the returned data. Indexed by rid.
		vector< offNodeData > offNodeData_;
		/// Manages the available Rids.
		vector< unsigned int > freeRidStack_;
};

extern const Cinfo* initShellCinfo();

/**
 * Do the whole operation of a blocking remote call, typically with a
 * return value.
 * Returns true if success.
 *   Shell is a ptr to the Shell object.
 *   Value is a pointer to the locally allocated data storage to be filled.
 *   	It can and should be a local variable, for thread safety.
 *   Arg is the data to be sent to the remote node.
 *   Slot identifies target message
 *   offNode identifies target node. If it is numNodes, this is handled
 *      as a broadcast request going to all nodes.
 *
 * This function requires two handlers (MsgDest) to be defined:
 * One for the remote shell, to deal with the request and return the
 * value, and one for the local shell, to deal with the return value.
 * This function requires two MsgSrcs to be defined: One for the requesting
 * Slot, and one for the return.
 *
 *   Naming:
 *      outgoing SrcFinfo: request<Value>Src
 *      return SrcFinfo:   return<Value>Src
 *      remote DestFinfo:  request<Value>
 *      return DestFinfo:  return<Value>
 *      outgoing Slot:     request<Value>Slot
 *      return Slot:       return<Value>Slot
 *
 * Restrictions on use:
 * 1. In single-threaded version, this call can only be used at Setup time.
 *    At runtime it would lead to node asynchrony.
 * 2. In both single and multi-threaded versions, the value must be local
 *    or otherwise unique storage.
 * 3. In both versions, the recv buffer has to be stacked if there are any
 *    further operations pending. Otherwise the return value overwrites the
 *    buffer.
 */
template< class T, class A > void getOffNodeValue( 
	Eref shellE, Slot slot, unsigned int offNode,
	T* value, const A& arg )
{
#ifndef USE_MPI // If serial
	assert( 0 );
#else
	Shell* sh = static_cast< Shell* >( shellE.data() );
	if ( offNode >= sh->numNodes() ) {
		unsigned int requestId = 
			sh->openOffNodeValueRequestInner( 
				static_cast< void* >( value ), sh->numNodes() - 1 );
		send2< A, unsigned int >( shellE, slot, arg, requestId );

		T* temp = static_cast< T* >( 
			sh->closeOffNodeValueRequestInner( requestId ) );
		assert( value == temp );
	} else {
		if ( offNode > sh->myNode() )
			offNode--;
		unsigned int requestId = 
			sh->openOffNodeValueRequestInner( 
				static_cast< void* >( value ), 1 );
		sendTo2< A, unsigned int >( shellE, slot, offNode, arg, requestId );

		T* temp = static_cast< T* >( 
			sh->closeOffNodeValueRequestInner( requestId ) );
		assert( value == temp );
	}
#endif // USE_MPI
}


/** 
 * Here again are the two matching functions for opening and closing
 * the request for an off-node value.
 *
 * requestOffNodeId
 * returns a thread-safe unique id for the request, so that 
 * we can examine the return values for ones we are interested in. 
 * This rid is an index to a vector of ints that counts pending 
 * returns on this id.
 * Returns the next free Rid and initializes returnedData_ entry.
 */
template< class T > unsigned int openOffNodeValueRequest( 
	Shell* sh, T* init, unsigned int numPending )
{
#ifndef USE_MPI // If serial
	assert( 0 );
	return 0;
#else
	return sh->openOffNodeValueRequestInner( 
		static_cast< void* >( init ), numPending );
#endif // USE_MPI
}


/**
 * Polls postmaster, converts and returns data stored at rid
 */
template< class T > T* closeOffNodeValueRequest( 
	Shell* sh, unsigned int rid )
{
#ifndef USE_MPI // If serial
	assert( 0 );
	return 0;
#else
	void* temp = sh->closeOffNodeValueRequestInner( rid );
	return static_cast< T* >( temp );
#endif // USE_MPI
}


/**
 * Gets the value pointer for the specifid rid. Typed wrapper for
 * shell function.
 */
template< class T > T* getOffNodeValuePtr( Shell* sh, unsigned int rid )
{
#ifndef USE_MPI // If serial
	assert( 0 );
	return 0;
#else
	// should have been initialized to a memory location to hold the value
	return static_cast< T* >( sh->getOffNodeValuePtrInner( rid ) );
#endif // USE_MPI
}

#endif // _SHELL_H
