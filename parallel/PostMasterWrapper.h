/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2006 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _PostMasterWrapper_h
#define _PostMasterWrapper_h
class PostMasterWrapper: 
	public PostMaster, public Neutral
{
	friend Element* processConnPostMasterLookup( const Conn* );
	friend Element* remoteCommandConnPostMasterLookup( const Conn* );
    public:
		PostMasterWrapper(const string& n)
		:
			Neutral( n ),
			srcSrc_( &srcOutConn_ ),
			remoteCommandSrc_( &remoteCommandConn_ ),
			pollRecvSrc_( &parProcessConn_ ),
			// processConn uses a templated lookup function,
			// remoteCommandConn uses a templated lookup function,
			parProcessConn_( this ),
			srcOutConn_( this ),
			destInConn_( this )
		{
			vector< unsigned long > segments(1,4);
			destInConn_.resize( segments );
		}
///////////////////////////////////////////////////////
//    Field header definitions.                      //
///////////////////////////////////////////////////////
		static int getMyNode( const Element* e ) {
			return static_cast< const PostMasterWrapper* >( e )->myNode_;
		}
///////////////////////////////////////////////////////
//    EvalField header definitions.                  //
///////////////////////////////////////////////////////
		int localGetPollFlag() const;
		static int getPollFlag( const Element* e ) {
			return static_cast< const PostMasterWrapper* >( e )->
			localGetPollFlag();
		}
		void localSetPollFlag( int value );
		static void setPollFlag( Conn* c, int value ) {
			static_cast< PostMasterWrapper* >( c->parent() )->
			localSetPollFlag( value );
		}
		int localGetRemoteNode() const;
		static int getRemoteNode( const Element* e ) {
			return static_cast< const PostMasterWrapper* >( e )->
			localGetRemoteNode();
		}
		void localSetRemoteNode( int value );
		static void setRemoteNode( Conn* c, int value ) {
			static_cast< PostMasterWrapper* >( c->parent() )->
			localSetRemoteNode( value );
		}
///////////////////////////////////////////////////////
// Msgsrc header definitions .                       //
///////////////////////////////////////////////////////
		static NMsgSrc* getSrcSrc( Element* e ) {
			return &( static_cast< PostMasterWrapper* >( e )->srcSrc_ );
		}

		static SingleMsgSrc* getRemoteCommandSrc( Element* e ) {
			return &( static_cast< PostMasterWrapper* >( e )->remoteCommandSrc_ );
		}

		static NMsgSrc* getPollRecvSrc( Element* e ) {
			return &( static_cast< PostMasterWrapper* >( e )->pollRecvSrc_ );
		}

///////////////////////////////////////////////////////
// dest header definitions .                         //
///////////////////////////////////////////////////////
		void destFuncLocal( long index ) {
		}
		static void destFunc( Conn* c ) {
			static_cast< PostMasterWrapper* >( c->parent() )->
				destFuncLocal(
				static_cast< SolverConn* >( c )->index() );
		}

		void ordinalFuncLocal( int tick );
		static void ordinalFunc( Conn* c, int tick ) {
			static_cast< PostMasterWrapper* >( c->parent() )->
				ordinalFuncLocal( tick );
		}

		void asyncFuncLocal( int tick ) {
			checkPendingRequests();
		}
		static void asyncFunc( Conn* c, int tick ) {
			static_cast< PostMasterWrapper* >( c->parent() )->
				asyncFuncLocal( tick );
		}

		void postIrecvFuncLocal( int tick );
		static void postIrecvFunc( Conn* c, int tick ) {
			static_cast< PostMasterWrapper* >( c->parent() )->
				postIrecvFuncLocal( tick );
		}

		void postSendFuncLocal( int tick );
		static void postSendFunc( Conn* c, int tick ) {
			static_cast< PostMasterWrapper* >( c->parent() )->
				postSendFuncLocal( tick );
		}

		void pollRecvFuncLocal( int tick );
		static void pollRecvFunc( Conn* c, int tick ) {
			static_cast< PostMasterWrapper* >( c->parent() )->
				pollRecvFuncLocal( tick );
		}

		void processFuncLocal( ProcInfo info ) {
			checkPendingRequests();
		}
		static void processFunc( Conn* c, ProcInfo info ) {
			static_cast< PostMasterWrapper* >( c->parent() )->
				processFuncLocal( info );
		}

		void reinitFuncLocal(  );
		static void reinitFunc( Conn* c ) {
			static_cast< PostMasterWrapper* >( c->parent() )->
				reinitFuncLocal(  );
		}

		void remoteCommandFuncLocal( string data );
		static void remoteCommandFunc( Conn* c, string data ) {
			static_cast< PostMasterWrapper* >( c->parent() )->
				remoteCommandFuncLocal( data );
		}

		void addOutgoingFuncLocal( Field src, int tick, int size );
		static void addOutgoingFunc( Conn* c, Field src, int tick, int size ) {
			static_cast< PostMasterWrapper* >( c->parent() )->
				addOutgoingFuncLocal( src, tick, size );
		}

		void addIncomingFuncLocal( Field dest, int tick, int size );
		static void addIncomingFunc( Conn* c, Field dest, int tick, int size ) {
			static_cast< PostMasterWrapper* >( c->parent() )->
				addIncomingFuncLocal( dest, tick, size );
		}


///////////////////////////////////////////////////////
// Synapse creation and info access functions.       //
///////////////////////////////////////////////////////

///////////////////////////////////////////////////////
// Conn access functions.                            //
///////////////////////////////////////////////////////
		static Conn* getProcessConn( Element* e ) {
			return &( static_cast< PostMasterWrapper* >( e )->processConn_ );
		}
		static Conn* getRemoteCommandConn( Element* e ) {
			return &( static_cast< PostMasterWrapper* >( e )->remoteCommandConn_ );
		}
		static Conn* getParProcessConn( Element* e ) {
			return &( static_cast< PostMasterWrapper* >( e )->parProcessConn_ );
		}
		static Conn* getSrcOutConn( Element* e ) {
			return &( static_cast< PostMasterWrapper* >( e )->srcOutConn_ );
		}
		static Conn* getDestInConn( Element* e ) {
			return &( static_cast< PostMasterWrapper* >( e )->destInConn_ );
		}

///////////////////////////////////////////////////////
// Class creation and info access functions.         //
///////////////////////////////////////////////////////
		static Element* create(
			const string& name, Element* pa, const Element* proto ) {
			// Put tests for parent class here
			// Put proto initialization stuff here
			// const PostMaster* p = dynamic_cast<const PostMaster *>(proto);
			// if (p)... and so on. 
			return new PostMasterWrapper(name);
		}

		const Cinfo* cinfo() const {
			return &cinfo_;
		}

		char* getPostPtr( unsigned long index );

    private:
///////////////////////////////////////////////////////
// MsgSrc template definitions.                      //
///////////////////////////////////////////////////////
		ParallelMsgSrc srcSrc_;
		SingleMsgSrc1< string > remoteCommandSrc_;
		NMsgSrc0 pollRecvSrc_;
		UniConn< processConnPostMasterLookup > processConn_;
		UniConn< remoteCommandConnPostMasterLookup > remoteCommandConn_;
		MultiConn parProcessConn_;
		MultiConn srcOutConn_;
		SolveMultiConn destInConn_;

///////////////////////////////////////////////////////
// Synapse definition.                               //
///////////////////////////////////////////////////////

///////////////////////////////////////////////////////
// Private functions and fields for the Wrapper class//
///////////////////////////////////////////////////////
		// void informTargetNode();
		void connectTick( Element* tick );
		bool callsMe( Element* tickElm );
		bool connect( const string& target );
		void checkPendingRequests();
		// void assignIncomingSizes();
		// void assignIncomingSchedule();
		void countTicks(); // Allocated arrays based on # of ticks.

///////////////////////////////////////////////////////
// Static initializers for class and field info      //
///////////////////////////////////////////////////////
		static Finfo* fieldArray_[];
		static const Cinfo cinfo_;
};
#endif // _PostMasterWrapper_h
