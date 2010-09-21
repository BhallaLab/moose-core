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

class ThreadInfo;

enum AssignmentType { SINGLE, VECTOR, REPEAT };

class Shell
{
	public:
		Shell();

		///////////////////////////////////////////////////////////
		// Field functions
		///////////////////////////////////////////////////////////
		void setName( string name );
		string getName() const;
		void setQuit( bool val );
		bool getQuit() const;
		void setCwe( Id cwe );
		Id getCwe() const;

		///////////////////////////////////////////////////////////
		// Parser functions
		///////////////////////////////////////////////////////////
		Id doCreate( string type, Id parent, string name, 
			vector< unsigned int > dimensions );

		bool doDelete( Id i );

		/**
		 * Sets up a Message of specified type.
		 * Later need to consider doing this through MsgSpecs only.
		 * Here the 'args' vector handles whatever arguments we may need
		 * to pass to the specified msgType.
		 */
		MsgId doAddMsg( const string& msgType, 
			FullId src, const string& srcField, 
			FullId dest, const string& destField);

		void doQuit( );

		/**
		 * Starts off simulation
		 */
		void doStart( double runtime );

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
		 * shifts orig Element to newParent.
		 */
		void doMove( Id orig, Id newParent );

		/**
		 * Copies orig Element to newParent. n specifies how many copies
		 * are made.
		 * copyExtMsgs specifies whether to also copy messages from orig
		 * to objects outside the copy tree. Usually we don't do this.
		 */
		Id doCopy( Id orig, Id newParent, string newName,
			unsigned int n, bool copyExtMsgs);

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

		///////////////////////////////////////////////////////////
		// DestFinfo functions
		///////////////////////////////////////////////////////////
		void handleGet( const Eref& e, const Qinfo* q, const char* arg );

		/**
		 * Sets of a simulation for duration runTime. Handles
		 * cases including single-thread, multithread, and multinode
		 */
		void start( double runTime );

		/**
		 * Wrapper for the start function. This adds the call back
		 * to acknowledge completion of op.
		 */
		void handleStart( double runTime );
		void handleReinit();
		void handleStop();
		void handleTerminate();

		void initAck();
		void handleAck( unsigned int ackNode, unsigned int status );
		bool isAckPending() const;

		void handleQuit();

		void handleCreate( const Eref& e, const Qinfo* q, 
			string type, Id parent, Id newElm, string name,
			vector< unsigned int > dimensions );
		void destroy( const Eref& e, const Qinfo* q, Id eid);
		void innerCreate( string type, Id parent, Id newElm, string name,
			const vector< unsigned int >& dimensions );

		// void addmsg( Id src, Id dest, string srcfield, string destfield );
		/**
		 * Connects src to dest on appropriate fields, with specified
		 * msgType. 
		 * This inner function does NOT send an ack. Returns true on 
		 * success
		 */
		bool innerAddMsg( string msgType, 
			FullId src, string srcField, 
			FullId dest, string destField);

		/**
		 * Connects src to dest on appropriate fields, with specified
		 * msgType. 
		 * This wrapper function sends the ack back to the master node.
		 */
		void handleAddMsg( string msgType, 
			FullId src, string srcField, 
			FullId dest, string destField);

		/**
		 * Moves Element orig onto the newParent.
		 */
		void handleMove( Id orig, Id newParent );

		/**
		 * Deep copy of source element to target, renaming it to newName.
		 * The Args are orig, newParent, newElm
		 * where the newElm is the Id passed in for the root of the copy.
		 * All subsequent created Elements should have successive Ids.
		 * The copy may generate an array with n entries.
		 * Normally only copies msgs within the tree, but if the flag
		 * copyExtMsgs is true then it copies external Msgs too.
		 */
		void handleCopy( vector< Id > args, string newName, unsigned int n, 
			bool copyExtMsgs );

		/**
		 * Sets up scheduling for elements on the path.
		 */
		void handleUseClock( string path, string field, unsigned int tick );

		////////////////////////////////////////////////////////////////
		// Thread and MPI handling functions
		////////////////////////////////////////////////////////////////

		void process( const Eref& e, ProcPtr p );
		
		/**
		 * Sets up master message that interconnects all shells on all
		 * nodes
		 */
		static void connectMasterMsg();
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

		void initThreadInfo( vector< ThreadInfo >& ti,
		Element* clocke, Qinfo* q,
		pthread_mutex_t* sortMutex, double runtime );

		/**
		 * Stub for eventual function to handle load balancing. This must
		 * be called to set up default groups.
		 */
		void loadBalance();

		/**
		 * Function to execute on the mpiThread. Deals with
		 * all MPI transactions. I am keeping it on a single thread
		 * because different MPI implementations vary in their thread-
		 * safety.
		 */
		static void* mpiThreadFunc( void* shellPtr );

		void launchParser();

		void launchMsgLoop( Element* shelle );
		/**
		 * Thread func for handling msgs.
		 */
		void* msgLoop( void* shelle );

		void passThroughMsgQs( Element* shelle );


		void setRunning( bool value );

		////////////////////////////////////////////////////////////////
		// Functions for handling field Set/Get operations
		////////////////////////////////////////////////////////////////


		/**
		 * Local node function to assign a single entry in target object
		 */
		void innerSet( const Eref& er, FuncId fid, const char* args,
			unsigned int size );

		/**
		 * Local node function to assign all entries in an array target
		 * object. The target object may be decomposed between nodes,
		 * this function has to figure out which part of the array to
		 * use for which entry.
		 */
		void innerSetVec( const Eref& er, FuncId fid, 
			const PrepackedBuffer& arg );


		/**
		 * Local node function that orchestrates the assignment. It picks
		 * the assignment mode to operate the appropriate innerSet 
		 * function.
		 */
		void handleSet( Id id, DataId d, FuncId fid, PrepackedBuffer arg );

		static void dispatchSet( const Eref& er, FuncId fid, 
			const char* args, unsigned int size );

		static void dispatchSetVec( const Eref& er, FuncId fid, 
			const PrepackedBuffer& arg );

		void innerDispatchSet( Eref& sheller, const Eref& er, 
			FuncId fid, const PrepackedBuffer& arg );

		static const char* dispatchGet( 
			const Eref& tgt, const string& field, const SetGet* sg );

		const char* innerDispatchGet( 
			const Eref& sheller, const Eref& tgt, FuncId tgtFid );

		void handleGet( Id id, DataId index, FuncId fid );

		void recvGet( unsigned int node, unsigned int status, 
			PrepackedBuffer pb );

		void lowLevelRecvGet( PrepackedBuffer pb );

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

		const char* getBuf() const;
		static const char* buf();
		static const ProcInfo* procInfo();

		static bool chopPath( const string& path, vector< string >& ret,
			char separator = '/' );

		static void wildcard( const string& path, vector< Id >& list );
	private:
		string name_;
		Element* shelle_; // It is useful for the Shell to have this.
		vector< char > getBuf_;
		MsgId latestMsgId_; // Hack to communicate newly made MsgIds.
		bool quit_;
		bool isSingleThreaded_;
		static unsigned int numCores_;
		static unsigned int numNodes_;
		static unsigned int myNode_;
		static ProcInfo p_; 
			// Shell owns its own ProcInfo, has global thread/node info.
			// Used to talk to parser and for thread specification in
			// setup operations.
		unsigned int numAcks_;
		vector< unsigned int > acked_;
		void* barrier1_;
		void* barrier2_;
		/**
		 * Used to coordinate threads especially when doing MPI.
		 */
		bool isRunning_;

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
};

/*
extern bool set( Eref& dest, const string& destField, const string& val );

extern bool get( const Eref& dest, const string& destField );
*/

#endif // _SHELL_H
