/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2005 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _RETURN_FINFO_H
#define _RETURN_FINFO_H

///////////////////////////////////////////////////////////////////////
// Here we define a set of classes derived from Finfo which
// handle situations where multiple incoming message calls each
// require immediate return calls to the originating object, without
// impinging on any other object. This means that the traditional
// send() call will not work for the return call, because send() is
// designed to send info to all the target messages.
// Example case: Multiple objects using the same lookup table to do
// some uniform kind of lookup. Happens with HHChannels and HHGates.
// 
// General implementation: ReturnFinfos manage a vector of ReturnConns
// directly. Each ReturnConn holds parent, target, and recvFunc.
// So the ReturnFinfo does not need to refer to a MsgSrc.
//
///////////////////////////////////////////////////////////////////////

// This is the base ReturnFinfo class. 
class ReturnFinfo: public Finfo
{
	public:
		ReturnFinfo( const string& name, 
			MultiReturnConn* ( *getConn )( Element * ),
			const string& triggers,
			bool sharesConn = 0
		)
		:	Finfo(name), 
			getConn_( getConn ),
			triggers_( triggers ),
			sharesConn_( sharesConn )
		{
			;
		}

		RecvFunc targetFunc( Element* e, unsigned long i ) const {
			return getConn_( e )->targetFunc( i ) ;
		}

		RecvFunc targetFuncFromSlot( Element* e, unsigned long slot )
		const {
			// This does not make sense
			cerr << "Error: Should not use ReturnFinfo::targetFuncFromSlot\n";
			return 0;
		}

		// Returns index of first match
		unsigned long indexOfMatchingFunc( Element* e, 
			RecvFunc rf ) const
		{
			return getConn_( e )->indexOfMatchingFunc( rf );
		}

		//returns number of matches
		unsigned long matchRemoteFunc( Element* e, RecvFunc rf ) const
		{
			return getConn_( e )->matchRemoteFunc( rf );
		}

		void addRecvFunc( Element* e, RecvFunc rf,
			unsigned long position ) {
			getConn_( e )->addRecvFunc( rf );
		}

		Conn* inConn( Element* e ) const {
			return getConn_( e );
		}

		Conn* outConn( Element* e ) const {
			return getConn_( e );
		}

		// The Conn targets are both src and dest.
		void src( vector< Field >& list, Element* e );

		// The Conn targets are both src and dest.
		void dest( vector< Field >& list, Element* e ) {
			src( list, e );
		}

		bool add( Element* e, Field& destfield, bool useSharedConn = 0);

		bool funcDrop( Element* e, const Conn* c ) {
			cerr << "ReturnFinfo::funcDrop called\n";
			return 0;
		}

		void initialize( const Cinfo* c ) {
			parseTriggerList( c, triggers_, internalSrc_ );
		}

		Finfo* respondToAdd( Element* e, const Finfo* sender );

		bool strGet( Element* e, string& val );

		private:
			MultiReturnConn* ( *getConn_ )( Element * );
			string triggers_;
			bool sharesConn_;
			vector< Finfo* > internalSrc_;
};

// This class handles multi-target messages with zero arguments.
class Return0Finfo: public ReturnFinfo
{
	public:
		Return0Finfo( const string& name, 
			MultiReturnConn* ( *getConn )( Element * ),
			const string& triggers,
			bool sharesConn = 0
		)
		:	ReturnFinfo( name, getConn, triggers, sharesConn ) 
		{
			;
		}

		// Used when an external input is to trigger a Send call.
		// Here the Conn will have to provide the extra info, the
		// specific field ptr, needed to execute this function.
		static void relayFunc( Conn* c );

		Finfo* makeRelayFinfo( Element* e ) {
			Field temp( this, e );
			return new RelayFinfo0( temp );
		}

		RecvFunc recvFunc() const {
			return relayFunc;
		}

		// Finfo* respondToAdd( Element* e, const Finfo* sender );
	
		const Ftype* ftype() const {
			static const Ftype0 myFtype_;
			return &myFtype_;
		}
};

// This class handles multi-target messages with one argument.
template <class T> class Return1Finfo: public ReturnFinfo
{
	public:
		Return1Finfo( const string& name, 
			MultiReturnConn* ( *getConn )( Element * ),
			const string& triggers,
			bool sharesConn = 0
		)
		:	ReturnFinfo( name, getConn, triggers, sharesConn ) 
		{
			;
		}

		// Used when an external input is to trigger a Send call.
		// Here the Conn will have to provide the extra info, the
		// specific field ptr, needed to execute this function.
		static void relayFunc( Conn* c, T v ) {
			RelayConn* cr = dynamic_cast< RelayConn* >( c );
			if (cr) {
				RelayFinfo1<T> *rf = 
					dynamic_cast< RelayFinfo1<T> *>( cr->finfo() );
				Return1Finfo<T> *f = 
					dynamic_cast< Return1Finfo<T> *>( rf->innerFinfo() );
				if (rf && f) {
					// Could call the target recvfunc here, but
					// it would be an unexpected behaviour. So
					// instead I complain.
					/*
					NMsgSrc1< T >* src = 
						dynamic_cast< NMsgSrc1< T >* >(
							f->getSrc_( c->parent() ) );
					if ( src )
						src->send( v );
						*/
					cerr << "Warning: Return1Finfo::relayFunc:: Probably don't want to send to all targets. Ignoring.\n";
				} else {
					cerr << "Error: Return1Finfo::relayFunc:: failed to cast Finfo\n";
				}
			} else {
				cerr << "Error: Return1Finfo::relayFunc:: failed to cast Conn\n";
			}
		}

		RecvFunc recvFunc() const {
			return reinterpret_cast< RecvFunc >( relayFunc );
		}

		RelayFinfo* makeRelayFinfo( Element* e ) {
			Field temp( this, e );
			return new RelayFinfo1< T >( temp );
		}
	
		const Ftype* ftype() const {
			static const Ftype1< T > myFtype_;
			return &myFtype_;
		}
};

// This class handles multi-target messages with two arguments.
template <class T1, class T2 > class Return2Finfo: public ReturnFinfo
{
	public:
		Return2Finfo( const string& name, 
			MultiReturnConn* ( *getConn )( Element * ),
			const string& triggers,
			bool sharesConn = 0
		)
		:	ReturnFinfo( name, getConn, triggers, sharesConn ) 
		{
			;
		}

		// Used when an external input is to trigger a Send call.
		// Here the Conn will have to provide the extra info, the
		// specific field ptr, needed to execute this function.
		static void relayFunc( Conn* c, T1 v1, T2 v2 ) {
			RelayConn* cr = dynamic_cast< RelayConn* >( c );
			if (cr) {
				RelayFinfo2< T1, T2 > *rf = 
					dynamic_cast< RelayFinfo2< T1, T2> *>( cr->finfo());
				Return2Finfo< T1, T2 > *f = 
					dynamic_cast< Return2Finfo<T1, T2> *>(
						rf->innerFinfo() );
				if (rf && f)
					// Could call the target recvfunc here, but
					// it would be an unexpected behaviour. So
					// instead I complain.
					/*
					reinterpret_cast< NMsgSrc2< T1, T2 >* >(
							f->getSrc_( c->parent() )
						)->send( v1, v2 );
					*/
					cerr << "Warning: Return2Finfo::relayFunc:: Probably don't want to send to all targets. Ignoring.\n";
				else
					cerr << "Error: Return2Finfo::relayFunc:: failed to cast Finfo\n";
			} else {
				cerr << "Error: Return2Finfo::relayFunc:: failed to cast Conn\n";
			}
		}

		RecvFunc recvFunc() const {
			return reinterpret_cast< RecvFunc >( relayFunc );
		}

		RelayFinfo* makeRelayFinfo( Element* e ) {
			Field temp( this, e );
			return new RelayFinfo2< T1, T2 >( temp );
		}
	
		const Ftype* ftype() const {
			static const Ftype2< T1, T2 > myFtype_;
			return &myFtype_;
		}
};

// This class handles multi-target messages with two arguments.
template <class T1, class T2, class T3 >
	class Return3Finfo: public ReturnFinfo
{
	public:
		Return3Finfo( const string& name, 
			MultiReturnConn* ( *getConn )( Element * ),
			const string& triggers,
			bool sharesConn = 0
		)
		:	ReturnFinfo( name, getConn, triggers, sharesConn ) 
		{
			;
		}

		// Used when an external input is to trigger a Send call.
		// Here the Conn will have to provide the extra info, the
		// specific field ptr, needed to execute this function.
		static void relayFunc( Conn* c, T1 v1, T2 v2, T3 v3 ) {
			RelayConn* cr = dynamic_cast< RelayConn* >( c );
			if (cr) {
				RelayFinfo3< T1, T2, T3 > *rf = 
					dynamic_cast< RelayFinfo3< T1, T2, T3> *>(
						cr->finfo());
				Return3Finfo< T1, T2, T3 > *f = 
					dynamic_cast< Return3Finfo<T1, T2, T3> *>(
						rf->innerFinfo() );
				if (rf && f)
					// Could call the target recvfunc here, but
					// it would be an unexpected behaviour. So
					// instead I complain.
					/*
					reinterpret_cast< NMsgSrc3< T1, T2, T3 >* >(
							f->getSrc_( c->parent() )
						)->send( v1, v2, v3 );
						*/
					cerr << "Warning: Return2Finfo::relayFunc:: Probably don't want to send to all targets. Ignoring.\n";
				else
					cerr << "Error: Return3Finfo::relayFunc:: failed to cast Finfo\n";
			} else {
				cerr << "Error: Return3Finfo::relayFunc:: failed to cast Conn\n";
			}
		}

		RecvFunc recvFunc() const {
			return reinterpret_cast< RecvFunc >( relayFunc );
		}

		RelayFinfo* makeRelayFinfo( Element* e ) {
			Field temp( this, e );
			return new RelayFinfo3< T1, T2, T3 >( temp );
		}
	
		const Ftype* ftype() const {
			static const Ftype3< T1, T2, T3 > myFtype_;
			return &myFtype_;
		}
};
#endif // _RETURN_FINFO_H
