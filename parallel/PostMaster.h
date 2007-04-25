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
		static unsigned int getMyNode( const Element* e );
		static unsigned int getRemoteNode( const Element* e );
		static void setRemoteNode( const Conn& c, unsigned int node );
		void* innerGetParBuf( unsigned int targetIndex,
						unsigned int size );
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
	private:
		unsigned int localNode_;
		unsigned int remoteNode_;
		char* inBuf_;
		unsigned int inBufSize_;
		vector< IncomingFunc > incomingFunc_;

		char* outBuf_;
		unsigned int outBufPos_;
		unsigned int outBufSize_;
		unsigned int outgoingSlotNum_;
		bool donePoll_;
		MPI::Request request_;
		MPI::Status status_;
		MPI::Comm* comm_;
};
#endif

#endif // _POST_MASTER_H
