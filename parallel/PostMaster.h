/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2006 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _POST_MASTER_H
#define _POST_MASTER_H

#ifdef USE_MPI

typedef void ( *TransferFunc )( Eref e, const char* );
class ParSyncMsgInfo
{
	public:
		ParSyncMsgInfo( TransferFunc tf )
			: tf_( tf )
		{;}

		void addTarget( Eref e )
		{
			proxies_.push_back( e );
		}

	private:
		TransferFunc tf_;
		vector< Eref > proxies_;
};

/**
 * A skeleton class for starting out the postmaster.
 */
class PostMaster
{
#ifdef DO_UNIT_TESTS
	friend void testParAsyncObj2Post();
	friend void testParAsyncObj2Post2Obj();
	friend void testShellSetupAsyncParMsg();
	friend void testBidirectionalParMsg();
	friend void testNodeSetup();
	friend void testParMsg();
#endif // DO_UNIT_TESTS
	public:
		PostMaster();
		//////////////////////////////////////////////////////////////
		// Field access functions
		//////////////////////////////////////////////////////////////
		/**
		 * Returns node on which this postmaster is running
		 */
		static unsigned int getMyNode( Eref e );

		/**
		 * Returns node with which this postmaster communicates
		 */
		static unsigned int getRemoteNode( Eref e );

		/**
		 * Assigns node with which to communicate
		 */
		static void setRemoteNode( const Conn* c, unsigned int node );

		/**
		 * Hack for handling shell-shell messages across nodes.
		 */
		static Id getShellProxy( Eref e );
		static void setShellProxy( const Conn* c, Id value );

		/**
		 * Hack for keeping track of number of incoming and outgoing async
		 * msgs.
		 */
		static void incrementNumAsyncIn( const Conn* c );
		static void incrementNumAsyncOut( const Conn* c );
		//////////////////////////////////////////////////////////////
		// Function to stuff data into async buffer. Used mostly
		// for testing.
		//////////////////////////////////////////////////////////////
		static void async( const Conn* c, char* data, unsigned int size );

		//////////////////////////////////////////////////////////////
		// Transmit/receive Data buffer handling functions.
		//////////////////////////////////////////////////////////////
		void* innerGetAsyncParBuf( const Conn* c, unsigned int size );
		/*
		void* innerGetParBuf( unsigned int targetIndex,
						unsigned int size );
		void* innerGetAsyncParBuf(
						unsigned int targetIndex, unsigned int size );
		void placeIncomingFuncs( 
					vector< IncomingFunc >&, unsigned int msgIndex );

		void parseMsgRequest( const char* req, Element* self );
					*/

	// Message handling
		static void postIrecv( const Conn* c );
		void innerPostIrecv();

		/**
		 * Wrapper for innerPoll
		 */
		static void poll( const Conn* c, bool doSync );
		/**
 		* This function does the main work of sending incoming messages
 		* to dests. It grinds through the irecv'ed data and sends it out
 		* to local dest objects.
 		* This is called by the poll function
 		*
 		* Does nothing if polling is complete (donePoll_ is true)
 		* If no message is expected:
 		* 	- Do single pass, test for message. Deal with it if needed
 		* 	  In any case, send back poll ack and set donePoll_ to 1
 		* If a message is expected:
 		*  - Do single pass, test for message, deal with it if needed.
 		*    - If message comes, deal with it, send back poll ack, 
		*      set donePoll_ = 1
 		* 	 - If message does not come, do nothing. Will need to poll later
 		*/
		void innerPoll( const Conn* c, bool doSync );
		static void postSend( const Conn* c, bool doSync );
		void innerPostSend( bool doSync );

		/**
		 * Handles synchronous data incoming in buffer.
		 */
		void handleSyncData();

		/**
		 * Handles asynchronous data incoming in buffer.
		 */
		void handleAsyncData();

		/**
		 * Executes an MPI::Barrier command if this is postmaster 0
		 */
		static void barrier( const Conn* c );
		void innerBarrier( );

		/**
 		* This blocking function works through the setup stack. It is where
 		* the system may recurse into further polling.
 		*/
		static void clearSetupStack( const Conn* c );
		void innerClearSetupStack( );

		/*
		void addIncomingFunc( unsigned int connId, unsigned int index );
		// This static function handles response to an addmsg request,
		// including operations on the element portion of the postmaster
		// and tranmitting info to the remote node.
		static unsigned int respondToAdd(
		Element* e, const string& respondString, unsigned int numDest );
		*/
	private:
		unsigned int localNode_;
		unsigned int remoteNode_;
		vector< char > sendBuf_;
		unsigned int sendBufPos_;
		unsigned int numSendBufMsgs_;

		
		/**
		 * Number of incoming async messages set up. On any given timestep,
		 * zero to numAsyncIn_ msgs may actually arrive.
		 */
		unsigned int numAsyncIn_; 

		/// Number of outgoing async messages set up.
		unsigned int numAsyncOut_;

		vector< char > recvBuf_;

		vector< ParSyncMsgInfo > syncInfo_;

		bool donePoll_;

		Id shellProxy_; // Hack for msgs between shells on different nodes.


		bool isRequestPending_; // True if request is pending.

		vector< vector< char > > setupStack_; // Used to manage pending setup data.
		MPI::Request request_;

		MPI::Status status_;
		const MPI::Comm* comm_;
};

class AsyncStruct {
#ifdef DO_UNIT_TESTS
	friend void testParAsyncObj2Post2Obj();
	friend void testShellSetupAsyncParMsg();
	friend void testBidirectionalParMsg();
#endif // DO_UNIT_TESTS
	public: 
		/**
		 * The target here actually refers to the proxy id, which is
		 * the source id on the originating node! Need to clean up.
		 */
		AsyncStruct( Id proxy, unsigned int funcIndex, unsigned int size )
			: proxy_( proxy ), funcIndex_( funcIndex ), size_( size )
		{;}

		AsyncStruct( const char* data )
		{ 
			// tgt_ = *( static_cast< const Id* >( data ) );
			proxy_ = *( const Id* ) ( data );
			data += sizeof( Id );
			funcIndex_ = *( const unsigned int* )( data );
			// tgtMsg_ = *( static_cast< const int* >( data ) );
			data += sizeof( unsigned int );
			size_ = *( const unsigned int* ) ( data );
			// srcMsg_ = *( static_cast< const int* >( data ) );
		}

		Id proxy() const {
			return proxy_;
		}

		int funcIndex() const {
			return funcIndex_;
		}

		unsigned int size() const {
			return size_;
		}

		void hackProxy( Id shellProxy ) {
			proxy_ = shellProxy;
		}

	private:
		Id proxy_;
		unsigned int funcIndex_;
		unsigned int size_;
};

extern const Cinfo* initPostMasterCinfo();
extern bool setupProxyMsg( 
			unsigned int srcNode, Id proxy, Id dest, int destMsg );

#endif // USE_MPI

#endif // _POST_MASTER_H
