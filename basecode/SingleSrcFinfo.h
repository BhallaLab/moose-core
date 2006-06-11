/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2005 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _SINGLE_SRC_FINFO_H
#define _SINGLE_SRC_FINFO_H

///////////////////////////////////////////////////////////////////////
// Here we define a set of classes derived from Finfo0 and Finfo1
// that handle messaging.
// Src0Finfo: Zero argument msgsrc
// Src1Finfo: One argument msgsrc
///////////////////////////////////////////////////////////////////////

void finfoErrorMsg( const string& name, Field dest );
void fillSrc( vector< Field >& list, Element* e, vector< Finfo* >& f );
extern bool parseShareList( 
	const Cinfo* c, const string&,
	vector< Finfo* >&, vector< Finfo* >&
);

// The name is just the name of the msgsrc
// The getSrc returns a ptr to the msgsrc, given the element ptr.
// The triggers are a list of msgdests that trigger this msgsrc.
// Here we just put in their finfo names.
class SingleSrcFinfo: public Finfo
{
	public:
		SingleSrcFinfo( const string& name, 
			SingleMsgSrc* ( *getSrc )( Element * ),
			const string& triggers,
			bool sharesConn
		)
		:	Finfo(name), 
			getSrc_( getSrc ),
			triggers_( triggers ),
			sharesConn_( sharesConn )
		{
			;
		}

		RecvFunc targetFunc( Element* e, unsigned long i ) const {
			return getSrc_( e )->targetFunc( i );
		}
		
		// returns number of matches
		unsigned long matchRemoteFunc( Element* e, RecvFunc rf ) const
		{
			return getSrc_( e )->matchRemoteFunc( rf );
		}

		void addRecvFunc( Element* e, RecvFunc rf, unsigned long position ) {
			getSrc_( e )->addRecvFunc( rf, position );
		}


		bool add( Element* e, Field& destfield, bool useSharedConn = 0);

		Conn* inConn( Element* e ) const {
			return dummyConn();
		}

		Conn* outConn( Element* e ) const {
			return  getSrc_( e )->conn();
		}

		void src( vector< Field >& list, Element* e ) {
			fillSrc( list, e, internalSrc_ );
		}

		bool funcDrop( Element* e, const Conn* c ) {
			return getSrc_( e )->funcDrop( c );
		}

		void initialize( const Cinfo* c ) {
			parseTriggerList( c, triggers_, internalSrc_ );
		}

		SingleMsgSrc* ( *getSrc_ )( Element * );

		Finfo* respondToAdd( Element* e, const Finfo* sender );

		bool strGet( Element* e, string& val );

		private:
			string triggers_;
			bool sharesConn_;
			vector< Finfo* > internalSrc_;
};

class SingleSrc0Finfo: public SingleSrcFinfo
{
	public:
		SingleSrc0Finfo( const string& name, 
			SingleMsgSrc* ( *getSrc )( Element * ),
			const string& triggers,
			bool sharesConn = 0
		)
		:	SingleSrcFinfo( name, getSrc, triggers, sharesConn )
		{
			;
		}

		// Used when an external input is to trigger a Send call.
		// Here the Conn will have to provide the extra info, the
		// specific field ptr, needed to execute this function.
		static void relayFunc( Conn* c );

		RecvFunc recvFunc() const {
			return relayFunc;
		}

		const Ftype* ftype() const {
			static const Ftype0 myFtype_;
			return &myFtype_;
		}

		Finfo* makeRelayFinfo( Element* e ) {
			Field temp( this, e );
			return new RelayFinfo0( temp );
		}
};

// The name is just the name of the msgsrc
// The getSrc returns a ptr to the msgsrc, given the element ptr.
// The triggers are a list of msgdests that trigger this msgsrc.
// Here we just put in their finfo names.
template <class T> class SingleSrc1Finfo: public SingleSrcFinfo
{
	public:
		SingleSrc1Finfo( const string& name, 
			SingleMsgSrc* ( *getSrc )( Element * ),
			const string& triggers,
			bool sharesConn = 0
		)
		:	SingleSrcFinfo( name, getSrc, triggers, sharesConn )
		{
			;
		}

		// Used when an external input is to trigger a Send call.
		// Here the Conn will have to provide the extra info, the
		// specific field ptr, needed to execute this function.
		static void relayFunc( Conn* c, T v ) {
			RelayConn* cr = dynamic_cast< RelayConn* >( c );
			if (cr) {
				SingleSrc1Finfo<T> *f = 
					dynamic_cast< SingleSrc1Finfo<T> *>( cr->finfo() );
				if (f)
					static_cast< SingleMsgSrc1< T >* >( 
						f->getSrc_( c->parent() )
					)->send( v );
			}
		}

		RecvFunc recvFunc() const {
			return reinterpret_cast< RecvFunc >( relayFunc );
		}
		
		const Ftype* ftype() const {
			static const Ftype1< T > myFtype_;
			return &myFtype_;
		}

		Finfo* makeRelayFinfo( Element* e ) {
			Field temp( this, e );
			return new RelayFinfo1< T >( temp );
		}
};

// The name is just the name of the msgsrc
// The getSrc returns a ptr to the msgsrc, given the element ptr.
// The triggers are a list of msgdests that trigger this msgsrc.
// Here we just put in their finfo names.
template <class T1, class T2> class SingleSrc2Finfo:
	public SingleSrcFinfo
{
	public:
		SingleSrc2Finfo( const string& name, 
			SingleMsgSrc* ( *getSrc )( Element * ),
			const string& triggers,
			bool sharesConn = 0
		)
		:	SingleSrcFinfo( name, getSrc, triggers, sharesConn )
		{
			;
		}

		// Used when an external input is to trigger a Send call.
		// Here the Conn will have to provide the extra info, the
		// specific field ptr, needed to execute this function.
		static void relayFunc( Conn* c, T1 v1, T2 v2 ) {
			RelayConn* cr = dynamic_cast< RelayConn* >( c );
			if (cr) {
				SingleSrc2Finfo< T1, T2 > *f = 
					dynamic_cast< SingleSrc2Finfo< T1, T2 > *>(
						cr->finfo() );
				if (f)
					static_cast< SingleMsgSrc2< T1, T2 >* >( 
						f->getSrc_( c->parent() )
					)->send( v1, v2 );
			}
		}

		RecvFunc recvFunc() const {
			return reinterpret_cast< RecvFunc >( relayFunc );
		}

		const Ftype* ftype() const {
			static const Ftype2< T1, T2 > myFtype_;
			return &myFtype_;
		}

		Finfo* makeRelayFinfo( Element* e ) {
			Field temp( this, e );
			return new RelayFinfo2< T1, T2 >( temp );
		}
};

// The name is just the name of the msgsrc
// The getSrc returns a ptr to the msgsrc, given the element ptr.
// The triggers are a list of msgdests that trigger this msgsrc.
// Here we just put in their finfo names.
template <class T1, class T2, class T3> class SingleSrc3Finfo:
	public SingleSrcFinfo
{
	public:
		SingleSrc3Finfo( const string& name, 
			SingleMsgSrc* ( *getSrc )( Element * ),
			const string& triggers,
			bool sharesConn = 0
		)
		:	SingleSrcFinfo( name, getSrc, triggers, sharesConn )
		{
			;
		}

		// Used when an external input is to trigger a Send call.
		// Here the Conn will have to provide the extra info, the
		// specific field ptr, needed to execute this function.
		static void relayFunc( Conn* c, T1 v1, T2 v2, T3 v3 ) {
			RelayConn* cr = dynamic_cast< RelayConn* >( c );
			if (cr) {
				SingleSrc3Finfo< T1, T2, T3 > *f = 
					dynamic_cast< SingleSrc3Finfo< T1, T2, T3 > *>(
						cr->finfo() );
				if (f)
					static_cast< SingleMsgSrc3< T1, T2, T3 >* >( 
						f->getSrc_( c->parent() )
					)->send( v1, v2, v3 );
			}
		}

		RecvFunc recvFunc() const {
			return reinterpret_cast< RecvFunc >( relayFunc );
		}

		const Ftype* ftype() const {
			static const Ftype3< T1, T2, T3 > myFtype_;
			return &myFtype_;
		}

		Finfo* makeRelayFinfo( Element* e ) {
			Field temp( this, e );
			return new RelayFinfo3< T1, T2, T3 >( temp );
		}
};
#endif // _SINGLE_SRC_FINFO_H
