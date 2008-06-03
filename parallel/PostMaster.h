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

		//////////////////////////////////////////////////////////////
		// Transmit/receive Data buffer handling functions.
		//////////////////////////////////////////////////////////////
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
		static void postIrecv( const Conn* c, int ordinal );
		void innerPostIrecv();
		static void poll( const Conn* c, int ordinal );
		void innerPoll( const Conn* c );
		static void postSend( const Conn* c, int ordinal );
		void innerPostSend( );

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

		vector< char > recvBuf_;

		vector< ParSyncMsgInfo > syncInfo_;

		bool donePoll_;
		
		MPI::Request request_;
		MPI::Status status_;
		MPI::Comm* comm_;
};

extern const Cinfo* initPostMasterCinfo();

#endif

#endif // _POST_MASTER_H
