/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2006 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

// This variant of ClockTick supports 5 stages involved in 
// managing parallel messaging.
//
// Stage 0: post irecv for this tick.
// Stage 1: Call all processes that have outgoing data on this tick.
// Stage 2: Post send
// Stage 3: Call all processes that only have local data.
// Stage 4: Poll for posted irecvs, as they arrive, send their contents.
//          The poll process relies on return info from each postmaster
//
// Stage 0, 2, 4 pass only tick stage info.
// Stage 1 and 3 pass regular ProcInfo

// Should really happen automatically when mpp sees it is derived.

#ifndef _ParTickWrapper_h
#define _ParTickWrapper_h
class ParTickWrapper: 
	public ParTick, public ClockTickWrapper
{
	friend Element* clockConnParTickLookup( const Conn* );
    public:
		ParTickWrapper(const string& n)
		:
			ClockTickWrapper( n ),
			outgoingProcessSrc_( &outgoingProcessConn_ ),
			outgoingReinitSrc_( &outgoingProcessConn_ ),
			ordinalSrc_( &parProcessConn_ ),
			asyncSrc_( &parProcessConn_ ),
			postIrecvSrc_( &parProcessConn_ ),
			postSendSrc_( &parProcessConn_ ),
			pollRecvSrc_( &parProcessConn_ ),
			// clockConn uses a templated lookup function,
			outgoingProcessConn_( this ),
			parProcessConn_( this ),
			pollAsyncInConn_( this )
		{
			;
		}
///////////////////////////////////////////////////////
//    Field header definitions.                      //
///////////////////////////////////////////////////////
		static void setHandleAsync( Conn* c, int value ) {
			static_cast< ParTickWrapper* >( c->parent() )->handleAsync_ = value;
		}
		static int getHandleAsync( const Element* e ) {
			return static_cast< const ParTickWrapper* >( e )->handleAsync_;
		}
///////////////////////////////////////////////////////
// Msgsrc header definitions .                       //
///////////////////////////////////////////////////////
		static NMsgSrc* getOutgoingProcessSrc( Element* e ) {
			return &( static_cast< ParTickWrapper* >( e )->outgoingProcessSrc_ );
		}

		static NMsgSrc* getOutgoingReinitSrc( Element* e ) {
			return &( static_cast< ParTickWrapper* >( e )->outgoingReinitSrc_ );
		}

		static NMsgSrc* getOrdinalSrc( Element* e ) {
			return &( static_cast< ParTickWrapper* >( e )->ordinalSrc_ );
		}

		static NMsgSrc* getAsyncSrc( Element* e ) {
			return &( static_cast< ParTickWrapper* >( e )->asyncSrc_ );
		}

		static NMsgSrc* getPostIrecvSrc( Element* e ) {
			return &( static_cast< ParTickWrapper* >( e )->postIrecvSrc_ );
		}

		static NMsgSrc* getPostSendSrc( Element* e ) {
			return &( static_cast< ParTickWrapper* >( e )->postSendSrc_ );
		}

		static NMsgSrc* getPollRecvSrc( Element* e ) {
			return &( static_cast< ParTickWrapper* >( e )->pollRecvSrc_ );
		}

///////////////////////////////////////////////////////
// dest header definitions .                         //
///////////////////////////////////////////////////////
		void pollRecvFuncLocal(  ) {
			++numArrived_;
		}
		static void pollRecvFunc( Conn* c ) {
			static_cast< ParTickWrapper* >( c->parent() )->
				pollRecvFuncLocal(  );
		}

		void processFuncLocal( ProcInfo info );
		static void processFunc( Conn* c, ProcInfo info ) {
			static_cast< ParTickWrapper* >( c->parent() )->
				processFuncLocal( info );
		}

		void reinitFuncLocal(  );
		static void reinitFunc( Conn* c ) {
			static_cast< ParTickWrapper* >( c->parent() )->
				reinitFuncLocal(  );
		}

		void pollAsyncFuncLocal(  ) {
			if ( handleAsync_ )
			asyncSrc_.send( ordinal() );
		}
		static void pollAsyncFunc( Conn* c ) {
			static_cast< ParTickWrapper* >( c->parent() )->
				pollAsyncFuncLocal(  );
		}


///////////////////////////////////////////////////////
// Synapse creation and info access functions.       //
///////////////////////////////////////////////////////

///////////////////////////////////////////////////////
// Conn access functions.                            //
///////////////////////////////////////////////////////
		static Conn* getClockConn( Element* e ) {
			return &( static_cast< ParTickWrapper* >( e )->clockConn_ );
		}
		static Conn* getOutgoingProcessConn( Element* e ) {
			return &( static_cast< ParTickWrapper* >( e )->outgoingProcessConn_ );
		}
		static Conn* getParProcessConn( Element* e ) {
			return &( static_cast< ParTickWrapper* >( e )->parProcessConn_ );
		}
		static Conn* getPollAsyncInConn( Element* e ) {
			return &( static_cast< ParTickWrapper* >( e )->pollAsyncInConn_ );
		}

///////////////////////////////////////////////////////
// Class creation and info access functions.         //
///////////////////////////////////////////////////////
		static Element* create(
			const string& name, Element* pa, const Element* proto );
		/*
		{
			// Put tests for parent class here
			// Put proto initialization stuff here
			// const ParTick* p = dynamic_cast<const ParTick *>(proto);
			// if (p)... and so on. 
			return new ParTickWrapper(name);
		}
		*/

		const Cinfo* cinfo() const {
			return &cinfo_;
		}


    private:
///////////////////////////////////////////////////////
// MsgSrc template definitions.                      //
///////////////////////////////////////////////////////
		NMsgSrc1< ProcInfo > outgoingProcessSrc_;
		NMsgSrc0 outgoingReinitSrc_;
		NMsgSrc1< int > ordinalSrc_;
		NMsgSrc1< int > asyncSrc_;
		NMsgSrc1< int > postIrecvSrc_;
		NMsgSrc1< int > postSendSrc_;
		NMsgSrc1< int > pollRecvSrc_;
		UniConn< clockConnParTickLookup > clockConn_;
		MultiConn outgoingProcessConn_;
		MultiConn parProcessConn_;
		PlainMultiConn pollAsyncInConn_;

///////////////////////////////////////////////////////
// Synapse definition.                               //
///////////////////////////////////////////////////////

///////////////////////////////////////////////////////
// Private functions and fields for the Wrapper class//
///////////////////////////////////////////////////////
		unsigned long numArrived_;
		unsigned long numPostMaster_;
		void separateOutgoingTargets();

///////////////////////////////////////////////////////
// Static initializers for class and field info      //
///////////////////////////////////////////////////////
		static Finfo* fieldArray_[];
		static const Cinfo cinfo_;
};
#endif // _ParTickWrapper_h
