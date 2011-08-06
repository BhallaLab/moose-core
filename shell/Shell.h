/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2009 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _SHELL_H
#define _SHELL_H

class FuncBarrier;
// class pthread_barrier_t;
// class pthread_mutex_t;
// class pthread_cond_t;
// class pthread_attr_t;

enum AssignmentType { SINGLE, VECTOR, REPEAT };
extern const ThreadId ScriptThreadNum; // Defined in Shell.cpp

extern SrcFinfo1< PrepackedBuffer >* lowLevelSetGet();
extern SrcFinfo4< Id, DataId, FuncId, PrepackedBuffer >* requestSet();
extern SrcFinfo4< Id, DataId, FuncId, unsigned int >* requestGet();
extern DestFinfo* receiveGet();
extern ReduceFinfoBase* reduceArraySizeFinfo();
extern SrcFinfo2< unsigned int, unsigned int >* ack();

class Shell
{
	public:
		Shell();
		~Shell();

		///////////////////////////////////////////////////////////
		// Field functions
		///////////////////////////////////////////////////////////
		/**
		 * Assigns the current working Element of the Shell
		 */
		void setCwe( Id cwe );

		/**
		 * Returns the current working Element of the Shell
		 */
		Id getCwe() const;

		/**
		 * Returns flag to indicate whether simulation is still running
		 */
		bool isRunning() const;

		///////////////////////////////////////////////////////////
		// Parser functions
		///////////////////////////////////////////////////////////

		/**
		 * Create an Element. Returns its id.
		 * type: Specifies classname of Objects in Element.
		 * parent: Id of parent element
		 * name: Name to be used for identifying Element.
		 * dimensions: Size of array in any # of dimensions.
		 */
		Id doCreate( string type, Id parent, string name, 
			vector< unsigned int > dimensions, bool isGlobal = 0 );

		/**
		 * Delete specified Element and all its children and all 
		 * Msgs connected to it.
		 */
		bool doDelete( Id id );

		/**
		 * Sets up a Message of specified type.
		 * Later need to consider doing this through MsgSpecs only.
		 * Here the 'args' vector handles whatever arguments we may need
		 * to pass to the specified msgType.
		 */
		MsgId doAddMsg( const string& msgType, 
			ObjId src, const string& srcField, 
			ObjId dest, const string& destField);

		/**
		 * Cleanly quits simulation, wrapping up all nodes and threads.
		 */
		void doQuit( );

		/**
		 * Starts off simulation, to run for 'runtime' more than current
		 * time. This version is blocking, and returns only when the 
		 * simulation is done.
		 */
		void doStart( double runtime );

		/**
		 * Starts off simulation, to run for 'runtime' more than current
		 * time. This version returns at once, and the parser can go
		 * on to do other things. It has to check with the 
		 * Shell::isRunning function (accessible as a MOOSE field)
		 * to find out if it is finished. Can call 'doStop', 'doTerminate'
		 * or 'doReinit' at any time to stop the run with increasing
		 * levels of prejudice.
		 */
		void doNonBlockingStart( double runtime );

		/**
		 * Reinitializes simulation: time goes to zero, all scheduled
		 * objects are set to initial conditions. If simulation is
		 * already running, first stops it.
		 */
		void doReinit();

		/**
		 * Cleanly stops simulation, ready to take up again from where
		 * the stop occurred. Waits till current operations are done.
		 */
		void doStop();

		/**
		 * Terminate ongoing simulation, with prejudice.
		 * Uncleanly stops simulation. Things may be in a mess with
		 * different objects at different times, but it stops at once.
		 */
		void doTerminate();

		/**
		 * shifts orig Element (including offspring) to newParent. All old 
		 * hierarchy, data, Msgs etc are preserved below the orig.
		 */
		void doMove( Id orig, Id newParent );

		/**
		 * Copies orig Element to newParent. n specifies how many copies
		 * are made.
		 * copyExtMsgs specifies whether to also copy messages from orig
		 * to objects outside the copy tree. Usually we don't do this.
		 */
		Id doCopy( Id orig, Id newParent, string newName,
			unsigned int n, bool toGlobal, bool copyExtMsgs);

		/**
		 * Looks up the Id specified by the given path. May include
		 * relative references and the internal cwe 
		 * (current working Element) on the shell
		 */
		Id doFind( const string& path ) const;

		/**
		 * Connects up process messages from the specified Tick to the
		 * targets on the path. Does so for whole Elements, not individual
		 * entries in the Element array.
		 * The target on the path usually has the 'process' field but
		 * other options are allowed, like 'init'
		 */
		void doUseClock( string path, string field, unsigned int tick );

		/**
		 * Loads in a model to a specified path.
		 * Tries to figure out model type from fname or contents of file.
		 * Currently knows about kkit,
		 * Soon to learn .p, SBML, NeuroML.
		 * Later to learn NineML
		 */
		 Id doLoadModel( const string& fname, const string& modelpath );

		/**
 		 * This function synchronizes fieldDimension on the DataHandler 
		 * across nodes. Used after function calls that might alter the
		 * number of Field entries in the table..
 		 * The tgt is the FieldElement whose fieldDimension needs updating.
 		 */
		void doSyncDataHandler( Id tgt );

		/**
 		 * This function builds a reac-diffusion mesh starting at the
 		 * specified ChemCompt, which houses MeshEntry FieldElements.
 		 * Assumes that the dimensions of the baseCompartment have just been
 		 * redefined, and we now need to go through and update the child 
 		 * reaction system.
 		 */
		void doReacDiffMesh( Id baseCompartment );

		/**
		 * Works through internal queue of operations that modify the
		 * structure of the simulation. These operations have to be 
		 * carefully separated from any other functions or messaging,
		 * so this happens while all other threads are blocked. 
		 */
		static void clearRestructuringQ();
		///////////////////////////////////////////////////////////
		// DestFinfo functions
		///////////////////////////////////////////////////////////
		// void handleGet( const Eref& e, const Qinfo* q, const char* arg );

		/**
		 * Sets of a simulation for duration runTime. Handles
		 * cases including single-thread, multithread, and multinode
		 */
		void start( double runTime );

		/**
 		 * Initialize acks. This call should be done before the 'send' goes 
 		 * out, because with the wonders of threading we might get a 
		 * response to the 'send' before this call is executed.
 		 * This MUST be followed by a waitForAck call.
 		 */
		void initAck();

		/**
 		 * test for completion of request. This MUST be preceded by an
 		 * initAck call.
 		 */
		void waitForAck();

		/**
 		 * Generic handler for ack msgs from various nodes. Keeps track of
 		 * which nodes have responded.
 		 */
		void handleAck( unsigned int ackNode, unsigned int status );

		/**
 		 * Test for receipt of acks from all nodes
 		 */ 
		bool isAckPending() const;

		void handleCreate( const Eref& e, const Qinfo* q, 
			string type, Id parent, Id newElm, string name,
			vector< unsigned int > dimensions );
		void destroy( const Eref& e, const Qinfo* q, Id eid);
		void innerCreate( string type, Id parent, Id newElm, string name,
			const vector< unsigned int >& dimensions );

		/**
		 * Connects src to dest on appropriate fields, with specified
		 * msgType. 
		 * This inner function does NOT send an ack. Returns true on 
		 * success
		 */
		bool innerAddMsg( string msgType, MsgId mid,
			ObjId src, string srcField, 
			ObjId dest, string destField);

		/**
		 * Connects src to dest on appropriate fields, with specified
		 * msgType. 
		 * This wrapper function sends the ack back to the master node.
		 */
		void handleAddMsg( const Eref& e, const Qinfo* q,
			string msgType, MsgId mid,
			ObjId src, string srcField, 
			ObjId dest, string destField);

		/**
		 * Moves Element orig onto the newParent.
		 */
		void handleMove( const Eref& e, const Qinfo* q,
			Id orig, Id newParent );

		/**
		 * Handles sync of DataHandler indexing across nodes
		 */
		void handleSync( const Eref& e, const Qinfo* q, Id elm, FuncId fid);

		/**
		 * Deep copy of source element to target, renaming it to newName.
		 * The Args are orig, newParent, newElm
		 * where the newElm is the Id passed in for the root of the copy.
		 * All subsequent created Elements should have successive Ids.
		 * The copy may generate an array with n entries.
		 * Normally only copies msgs within the tree, but if the flag
		 * copyExtMsgs is true then it copies external Msgs too.
		 */
		void handleCopy( const Eref& e, const Qinfo* q,
			vector< Id > args, string newName, unsigned int n, 
			bool toGlobal, bool copyExtMsgs );

		/**
		 * Sets up scheduling for elements on the path.
		 */
		void handleUseClock( const Eref& e, const Qinfo* q,
			string path, string field, unsigned int tick );

		/**
 		 * Tell all attached pools and vols to update themselves: set their
 		 * array sizes and set their new volumes.
 		 */
		void handleReMesh( Id baseMesh );
		////////////////////////////////////////////////////////////////
		// Thread and MPI handling functions
		////////////////////////////////////////////////////////////////

		void process( const Eref& e, ProcPtr p );
		
		/**
		 * Sets up master message that interconnects all shells on all
		 * nodes, and other initialization of basic messages.
		 */
		void connectMasterMsg();
		/**
		 * Assigns the hardware availability. Assumes that each node will
		 * have the same number of cores available.
		 */
		void setHardware( bool isSingleThreaded, 
			unsigned int numCores, unsigned int numNodes, 
			unsigned int myNode );

		static unsigned int myNode();
		static unsigned int numNodes();
		static unsigned int numCores();

		/**
		 * Stub for eventual function to handle load balancing. This must
		 * be called to set up default groups.
		 */
		void loadBalance();

		void launchParser();

		/**
	 	 * True when the parser is in a call which is being blocked becaus
		 * it requires the event loop to complete some actions.
		 */
		bool inBlockingParserCall() const;

		/**
		 * True in single-threaded mode. This is a special mode of
		 * the system in which it does not start up the event loop
		 * at all, and the whole thing operates on one thread, which
		 * is ultimately under the control of the parser.
		 * Note that this is distinct from running on one core. It is
		 * possible, and even recommended, to run in multithread mode
		 * even when the system has just one core to run it on.
		 */
		bool isSingleThreaded() const;

		/**
		 * This function sets up the threading for the entire system.
		 * It creates all the worker threads and the threads for
		 * handling MPI and handling shell requests.
		 */
		void launchThreads();

		/**
		 * This function closes up shop with all the threads.
		 */
		void joinThreads();

		pthread_mutex_t* parserMutex() const; 

		pthread_cond_t* parserBlockCond() const; 

		/**
		 * Checks for highest 'val' on all nodes
		 */
		static unsigned int reduceInt( unsigned int val );

		////////////////////////////////////////////////////////////////
		// Functions for handling field Set/Get operations
		////////////////////////////////////////////////////////////////

		/**
		 * Local node function that orchestrates the assignment. It picks
		 * the assignment mode to operate the appropriate innerSet 
		 * function.
		void handleSet( const Eref& e, const Qinfo*q,
			Id id, DataId d, FuncId fid, PrepackedBuffer arg );

		static void dispatchSet( const ObjId& oid, FuncId fid, 
			const char* args, unsigned int size );

		static void dispatchSetVec( const ObjId& oid, FuncId fid, 
			const PrepackedBuffer& arg );

		void innerDispatchSet( Eref& sheller, const ObjId& oid, 
			FuncId fid, const PrepackedBuffer& arg );

		static const vector< double* >* awaitGottenData();
		 */
		/*
		const vector< char* >& dispatchGet(
			const Eref& sheller,
			const ObjId& tgt, FuncId fid,
			const PrepackedBuffer& buf );
			*/
		/*
		static const vector< char* >& dispatchGet( 
			const ObjId& tgt, const string& field, const SetGet* sg,
			unsigned int& numGetEntries );

		const vector< char* >& innerDispatchGet( 
			const Eref& sheller, const ObjId& tgt, FuncId tgtFid,
			unsigned int numGetEntries );

		void handleGet( const Eref& e, const Qinfo*q,
			Id id, DataId index, FuncId fid, unsigned int numTgt );
			*/

		void recvGet( const Eref& e, const Qinfo* q, PrepackedBuffer pb );

		////////////////////////////////////////////////////////////////
		// Sets up clock ticks. Essentially is a call into the 
		// Clock::setupTick function, but may be needed to be called from
		// the parser so it is a Shell function too.
		void doSetClock( unsigned int tickNum, double dt );

		// Should set these up as streams so that we can build error
		// messages similar to cout.
		void warning( const string& text );
		void error( const string& text );

		static const Cinfo* initCinfo();

		////////////////////////////////////////////////////////////////
		// Utility functions
		////////////////////////////////////////////////////////////////
		static bool adopt( Id parent, Id child );

		static const unsigned int OkStatus;
		static const unsigned int ErrorStatus;

		// Initialization function, used only in main.cpp:init()
		void setShellElement( Element* shelle );

		const vector< double* >& getBuf() const;
		void clearGetBuf();

		/// Static func for returning the ProcInfo of the shell.
		static const ProcInfo* procInfo();

		const ProcInfo* getProcInfo( unsigned int index ) const;

		/// Digests outcome of calculation for max index of ragged array
		void digestReduceFieldDimension( 
			const Eref& er, const ReduceFieldDimension* arg );

		/**
 		 * static func.
 		 * Chops up the names in the path into the vector of strings. 
 		 * Returns true if it starts at '/'.
 		 */
		static bool chopPath( const string& path, vector< string >& ret,
			char separator = '/' );

		static void wildcard( const string& path, vector< Id >& list );

		/**
		 * Cleans up all Elements except /root itself, /clock, /classes,
		 * and /Msgs.
		 * In due course will also do suitable reinitialization of
		 * tick and other values.
		 */
		 static void cleanSimulation();

		/**
		 * Clears out the three temporary Msgs that are used for 
		 * assignments: Reduce, AssignVec and Assignment
		 */
		void clearSetMsgs();

		/**
		 * Flag: True when the waitForGetAck has requested that the 
		 * ProcessLoop go around one more time to harvest pending queue
		 * entries from other threads.
		 */
		unsigned int anotherCycleFlag_;

		/**
		 * set the gettingVector_ flag 
		 */
		void expectVector( bool flag );
		
	private:
		Element* shelle_; // It is useful for the Shell to have this.

		/**
		 * Buffer into which return values from the 'get' command are placed
		 * Only index 0 is used for any single-value 'get' call.
		 * If it was the 'getVec' command then the array is filled up
		 */
		vector< double* > getBuf_;

		/**
		 * Flag, used by the 'get' subsystem which maintains a buffer for
		 * returned value. True when the returned value is a vector.
		 */
		bool gettingVector_;

		/**
		 * Counter, used by the 'get' subsystem in order to keep track of
		 * number of returned values, especially for getVec.
		 */
		unsigned int numGetVecReturns_;

		MsgId latestMsgId_; // Hack to communicate newly made MsgIds.

		/**
		 * Flag: True when system is operating in single threaded mode.
		 * Used primarily for unit tests.
		 */
		bool isSingleThreaded_;

		/**
		 * Flag: True when the parser thread is blocked waiting for 
		 * some system call to be handled by the threading and the 
		 * MPI connected nodes.
		 */
		bool isBlockedOnParser_;


		/**
		 * Number of CPU cores in system. Specifies how many working threads
		 * will be assigned. Additional threads are created for parser
		 * and graphics.
		 */
		static unsigned int numCores_;

		/**
		 * Number of nodes in MPI-based system. Each node may have many
		 * threads.
		 */
		static unsigned int numNodes_;

		/**
		 * Identifier for current node
		 */
		static unsigned int myNode_;

		/**
		 * Shell owns its own ProcInfo, has global thread/node info.
		 * Used to talk to parser and for thread specification in
		 * setup operations.
		 */
		static ProcInfo p_; 

		vector< ProcInfo > threadProcs_;

		/**
		 * Array of threads, initialized in launchThreads.
		 */
		pthread_t* threads_;
		pthread_t* gThread_; /// graphics thread.
		pthread_attr_t *attr_;

		unsigned int numAcks_;
		vector< unsigned int > acked_;
		FuncBarrier* barrier1_;
		FuncBarrier* barrier2_;
		FuncBarrier* barrier3_;

		/**
		 * Pthreads mutex for synchronizing parser calls with underlying
		 * thread work cycle. Void here so we can compile without pthreads.
		 */
		pthread_mutex_t* parserMutex_; 

		/**
		 * Pthreads conditional for synchronizing parser calls with 
		 * thread work cycle. Void here so we can compile without pthreads.
		 */
		pthread_cond_t* parserBlockCond_; 

		/**
		 * Flag to tell system to reinitialize. We use this to defer the
		 * actual operation to the 'process' call for a clean reinit.
		 */
		bool doReinit_;
		/**
		 * Simulation run time
		 */
		double runtime_;

		/// Current working Element
		Id cwe_;

		/// Return value from reduceMax calculation for array sizes.
		unsigned int maxIndex_;

		/// Keeps track of allocated assignmentMsg so we can deallocate.
		MsgId assignmentMsg_;

		/// Keeps track of allocated assignVecMsg so we can deallocate.
		MsgId assignVecMsg_;

		/// Keeps track of allocated reduceMsg so we can deallocate.
		MsgId reduceMsg_;
};

/*
extern bool set( Eref& dest, const string& destField, const string& val );

extern bool get( const Eref& dest, const string& destField );
*/

#endif // _SHELL_H
