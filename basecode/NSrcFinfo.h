/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2005 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _NSRC_FINFO_H
#define _NSRC_FINFO_H

///////////////////////////////////////////////////////////////////////
// Here we define a set of classes derived from Finfo0 and Finfo1
// that handle messaging.
// Src0Finfo: Zero argument msgsrc
// Src1Finfo: One argument msgsrc
///////////////////////////////////////////////////////////////////////

// This is the base NSrcFinfo class, used whenever there are multiple
// targets
class NSrcFinfo: public Finfo
{
	public:
		NSrcFinfo( const string& name, 
			NMsgSrc* ( *getSrc )( Element * ),
			const string& triggers,
			bool sharesConn = 0
		)
		:	Finfo(name), 
			getSrc_( getSrc ),
			triggers_( triggers ),
			sharesConn_ ( sharesConn )
		{
			;
		}

		RecvFunc targetFunc( Element* e, unsigned long i ) const {
			return getSrc_( e )->targetFunc( i );
		}

		RecvFunc targetFuncFromSlot( Element* e, unsigned long slot )
				const {
			return getSrc_( e )->targetFuncFromSlot( slot );
		}

		unsigned long nFuncs( Element* e ) const {
			return getSrc_( e )->nFuncs();
		}

		// Returns index of first match
		unsigned long indexOfMatchingFunc( Element* e, 
			RecvFunc rf ) const
		{
			return getSrc_( e )->indexOfMatchingFunc( rf );
		}

		//returns number of matches
		unsigned long matchRemoteFunc( Element* e, RecvFunc rf ) const
		{
			return getSrc_( e )->matchRemoteFunc( rf );
		}

		void addRecvFunc( Element* e, RecvFunc rf, unsigned long position ) {
			getSrc_( e )->addRecvFunc( rf, position );
		}

		Conn* inConn( Element* e ) const {
			// internalSrc_->setParent( e );
			// return internalSrc_;
			return dummyConn();
		}

		Conn* outConn( Element* e ) const {
			return getSrc_( e )->conn();
		}

		void src( vector< Field >& list, Element* e ) {
			fillSrc( list, e, internalSrc_ );
		}

		void dest( vector< Field >& list, Element* e ) {
			getSrc_( e )->dest( list );
		}

		bool add( Element* e, Field& destfield, bool useSharedConn = 0);
		// void resize( Element* e, vector< unsigned long >& segment );

		bool funcDrop( Element* e, const Conn* c ) {
			return getSrc_( e )->funcDrop( c );
		}

		void initialize( const Cinfo* c ) {
			parseTriggerList( c, triggers_, internalSrc_ );
		}

		NMsgSrc* ( *getSrc_ )( Element * );

		Finfo* respondToAdd( Element* e, const Finfo* sender );

		bool strGet( Element* e, string& val );

		private:
			string triggers_;
			bool sharesConn_;
			vector< Finfo* > internalSrc_;
};

// This class handles multi-target messages with zero arguments.
class NSrc0Finfo: public NSrcFinfo
{
	public:
		NSrc0Finfo( const string& name, 
			NMsgSrc* ( *getSrc )( Element * ),
			const string& triggers,
			bool sharesConn = 0
		)
		:	NSrcFinfo( name, getSrc, triggers, sharesConn ) 
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
template <class T> class NSrc1Finfo: public NSrcFinfo
{
	public:
		NSrc1Finfo( const string& name, 
			NMsgSrc* ( *getSrc )( Element * ),
			const string& triggers,
			bool sharesConn = 0
		)
		:	NSrcFinfo( name, getSrc, triggers, sharesConn ) 
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
				NSrc1Finfo<T> *f = 
					dynamic_cast< NSrc1Finfo<T> *>( rf->innerFinfo() );
				if (rf && f) {
					NMsgSrc1< T >* src = 
						reinterpret_cast< NMsgSrc1< T >* >(
							f->getSrc_( c->parent() ) );
					if ( src )
						src->send( v );
				} else {
					cerr << "Error: NSrc1Finfo::relayFunc:: failed to cast Finfo\n";
				}
			} else {
				cerr << "Error: NSrc1Finfo::relayFunc:: failed to cast Conn\n";
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
template <class T1, class T2 > class NSrc2Finfo: public NSrcFinfo
{
	public:
		NSrc2Finfo( const string& name, 
			NMsgSrc* ( *getSrc )( Element * ),
			const string& triggers,
			bool sharesConn = 0
		)
		:	NSrcFinfo( name, getSrc, triggers, sharesConn ) 
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
				NSrc2Finfo< T1, T2 > *f = 
					dynamic_cast< NSrc2Finfo<T1, T2> *>(
						rf->innerFinfo() );
				if (rf && f)
					reinterpret_cast< NMsgSrc2< T1, T2 >* >(
							f->getSrc_( c->parent() )
						)->send( v1, v2 );
				else
					cerr << "Error: NSrc2Finfo::relayFunc:: failed to cast Finfo\n";
			} else {
				cerr << "Error: NSrc2Finfo::relayFunc:: failed to cast Conn\n";
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
	class NSrc3Finfo: public NSrcFinfo
{
	public:
		NSrc3Finfo( const string& name, 
			NMsgSrc* ( *getSrc )( Element * ),
			const string& triggers,
			bool sharesConn = 0
		)
		:	NSrcFinfo( name, getSrc, triggers, sharesConn ) 
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
				NSrc3Finfo< T1, T2, T3 > *f = 
					dynamic_cast< NSrc3Finfo<T1, T2, T3> *>(
						rf->innerFinfo() );
				if (rf && f)
					reinterpret_cast< NMsgSrc3< T1, T2, T3 >* >(
							f->getSrc_( c->parent() )
						)->send( v1, v2, v3 );
				else
					cerr << "Error: NSrc2Finfo::relayFunc:: failed to cast Finfo\n";
			} else {
				cerr << "Error: NSrc2Finfo::relayFunc:: failed to cast Conn\n";
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
#endif // _NSRC_FINFO_H
