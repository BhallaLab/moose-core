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

class Shell: public Data
{
	public:
		Shell();
		void process( const ProcInfo* p, const Eref& e );
		void setName( string name );
		string getName() const;
		void setQuit( bool val );
		bool getQuit() const;

		///////////////////////////////////////////////////////////
		// Parser functions
		///////////////////////////////////////////////////////////
		Id doCreate( string type, Id parent, string name, 
			vector< unsigned int > dimensions );

		bool doDelete( Id i );

		/**
		 * Sets up a Message of specified type.
		 * Later need to consider doing this through MsgSpecs only.
		 */
		MsgId doCreateMsg( Id src, const string& srcField, Id dest,
			const string& destField, const string& msgType );

		///////////////////////////////////////////////////////////
		// DestFinfo functions
		///////////////////////////////////////////////////////////
		void handleGet( Eref e, const Qinfo* q, const char* arg );

		/**
		 * Sets of a simulation for duration runTime. Handles
		 * cases including single-thread, multithread, and multinode
		 */
		void start( double runTime );

		void handleAckCreate();
		void handleAckDelete();
	

		void create( Eref e, const Qinfo* q, 
			string type, Id parent, Id newElm, string name );
		void destroy( Eref e, const Qinfo* q, Id eid);
		void innerCreate( string type, Id parent, Id newElm, string name );
		void addmsg( Id src, Id dest, string srcfield, string destfield );

		const char* getBuf() const;
		static const char* buf();
		static const ProcInfo* procInfo();

		////////////////////////////////////////////////////////////////
		// Thread and MPI handling functions
		////////////////////////////////////////////////////////////////
		/**
		 * Assigns the hardware availability. Assumes that each node will
		 * have the same number of cores available.
		 */
		void setHardware( bool isSingleThreaded, 
			unsigned int numCores, unsigned int numNodes, 
			unsigned int myNode );

		unsigned int myNode() const ;
		unsigned int numCores() const;

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
		// Sets up clock ticks. Essentially is a call into the 
		// Clock::setupTick function, but may be needed to be called from
		// the parser so it is a Shell function too.
		void setclock( unsigned int tickNum, double dt, unsigned int stage );

		// Should set these up as streams so that we can build error
		// messages similar to cout.
		void warning( const string& text );
		void error( const string& text );

		static const Cinfo* initCinfo();
	private:
		string name_;
		vector< char > getBuf_;
		bool quit_;
		bool isSingleThreaded_;
		unsigned int numCores_;
		unsigned int numNodes_;
		unsigned int myNode_;
		static ProcInfo p_; 
			// Shell owns its own ProcInfo, has global thread/node info.
			// Used to talk to parser and for thread specification in
			// setup operations.
		unsigned short numCreateAcks_;
		unsigned short numDeleteAcks_;
		void* barrier_;
		/**
		 * Used to coordinate threads especially when doing MPI.
		 */
		bool isRunning_;
};

extern bool set( Eref& dest, const string& destField, const string& val );

extern bool get( const Eref& dest, const string& destField );

#endif // _SHELL_H
