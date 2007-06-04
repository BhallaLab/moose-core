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
		static unsigned int getMyNode( const Element* e );
		static unsigned int getRemoteNode( const Element* e );
		static void setRemoteNode( const Conn& c, unsigned int node );
		static unsigned int getTargetId( const Element* e );
		static void setTargetId( const Conn& c, unsigned int value );
		static string getTargetField( const Element* e );
		static void setTargetField( const Conn& c, string value );

		//////////////////////////////////////////////////////////////
		// Transmit/receive Data buffer handling functions.
		//////////////////////////////////////////////////////////////
		void* innerGetParBuf( unsigned int targetIndex,
						unsigned int size );
		void* innerGetAsyncParBuf(
						unsigned int targetIndex, unsigned int size );
		void placeIncomingFuncs( 
					vector< IncomingFunc >&, unsigned int msgIndex );

		void outgoingFunc( );
		void parseMsgRequest( const char* req, Element* self );
	// Message handling
		static void postIrecv( const Conn& c, int ordinal );
		void innerPostIrecv();
		static void poll( const Conn& c, int ordinal );
		void innerPoll( Element* e);
		static void postSend( const Conn& c, int ordinal );
		void innerPostSend( );

		void addIncomingFunc( unsigned int connId, unsigned int index );
		// This static function handles response to an addmsg request,
		// including operations on the element portion of the postmaster
		// and tranmitting info to the remote node.
		static unsigned int respondToAdd(
		Element* e, const string& respondString, unsigned int numDest );
	private:
		unsigned int localNode_;
		unsigned int remoteNode_;
		char* inBuf_;
		unsigned int inBufSize_;
		vector< unsigned int > incomingFunc_;

		char* outBuf_;
		unsigned int outBufPos_;
		unsigned int outBufSize_;
//		unsigned int outgoingSlotNum_;
		bool donePoll_;

		// Here are some fields used to ferry data into off-node
		// messaging code in ParFinfo.
		unsigned int targetId_;
		string targetField_;
		
		MPI::Request request_;
		MPI::Status status_;
		MPI::Comm* comm_;
};

extern const char* ftype2str( const Ftype *f );

#endif

#endif // _POST_MASTER_H
