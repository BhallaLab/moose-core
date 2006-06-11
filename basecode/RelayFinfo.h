/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2005 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _RELAY_FINFO_H
#define _RELAY_FINFO_H

///////////////////////////////////////////////////////////////////////
// Here we define a set of Finfo classes 
// that handle relaying.
///////////////////////////////////////////////////////////////////////
// For msgs from msgdests, or from fields, we have 4 options:
// - Execute outgoing msg after original op
// - Execute outgoing msg before original op
// - Execute outgoing msg instead of original op
// - just block original op
// How to select one of these?
// We default to the first. Execute original op, then do outgoing msg.

// Requests the element e to remove the relay Finfo f
extern void cleanRelayFinfos( Element* e, Finfo* f );

class RelayFinfo: public Finfo
{
	public:
		RelayFinfo( Field& f )
		:	Finfo( f->name() ),
			f_ (f),
			inConn_( f.getElement(), this ),
			outConn_( f.getElement(), this )
		{
			;
		}

		RecvFunc targetFunc( Element* e, unsigned long i ) const {
			if ( i < rfunc_.size() )
				return rfunc_[ i ];
			return 0;
		}

		unsigned long matchRemoteFunc(
			Element* e, RecvFunc rf ) const {
			return static_cast< unsigned long > (
				count( rfunc_.begin(), rfunc_.end(), rf ) );
		}

		void addRecvFunc( Element* e, RecvFunc rf,
			unsigned long position ) ;

		Conn* inConn( Element* ) const {
			return const_cast< RelayConn* >( &inConn_ );
		}

		Conn* outConn( Element* ) const {
			return const_cast< RelayConn* >( &outConn_ );
		}

		void src( vector< Field >& list, Element* e ) {
			Finfo::src( list, e );
			innerFinfo()->src( list, e );
		}

		void dest( vector< Field >& list, Element* e );
		bool add( Element* e, Field& destfield, bool useSharedConn = 0);
		Finfo* respondToAdd( Element* e, const Finfo* sender );

		void initialize( const Cinfo* c ) {
			;
		}

		Finfo* innerFinfo() {
			return f_.operator->();
		}

		bool dropRelay( unsigned long index,  Conn* src );

		// Cleans up if there is nothing to connect to.
		void handleEmptyConn( Conn* c );

		Finfo* makeRelayFinfo( Element* e ) {
			return this; // Not sure about this. 
		}

	protected:
		vector < RecvFunc > rfunc_;
	private:
		Field f_;
		RelayConn inConn_;
		RelayConn outConn_;
};

class RelayFinfo0: public RelayFinfo
{
	public:
		RelayFinfo0( Field& f )
		:	RelayFinfo( f )
		{
			;
		}

		// Used to intercept incoming msg calls.
		static void relayFunc( Conn* c );

		RecvFunc recvFunc() const {
			return relayFunc;
		}

		const Ftype* ftype() const  {
			static const Ftype0 myFtype_;
			return &myFtype_;
		}
};

template < class T > class RelayFinfo1: public RelayFinfo
{
	public:
		RelayFinfo1( Field& f )
		:	RelayFinfo( f )
		{
			;
		}

		// The call first gets sent down to the recvFunc of the
		// original field. Then it comes back up, doing
		// all the outgoing calls (if any) from this RelayFinfo.
		static void relayFunc( Conn* c, T v ) {
			RelayConn* cr = dynamic_cast< RelayConn* >( c );
			if (cr) {
				RelayFinfo1< T >* f = 
					dynamic_cast< RelayFinfo1< T > *>( cr->finfo() );
				if ( f ) {
					// Do the original recvfunc
					RecvFunc oldfunc = f->innerFinfo()->recvFunc();
					reinterpret_cast< void ( * )( Conn*, T ) >(oldfunc)
						( c, v );
					// Then do the outgoing stuff if any
					for (unsigned long i = 0;
						i < f->outConn( c->parent() )->nTargets(); i++){
						reinterpret_cast< void ( * )( Conn*, T ) >(
							f->rfunc_[ i ])( 
							f->outConn( c->parent() )->target( i ), v
						);
					}
				}
			}
		}

/*
		static void objLookupFunc( Conn* c, T v ) {
			RelayConn* cr = dynamic_cast< RelayConn* >( c );
			if (cr) {
				RelayFinfo1< T >* f = 
					dynamic_cast< RelayFinfo1< T > *>( cr->finfo() );
				if ( f ) {
					Element* pa = lookup_( c->parent(), f->index_ );
					if ( !Ftype1< T >::set( pa, f->finfo(), value ) ) {
						cerr << "Error::RelayFinfo1::ObjLookupFunc: Field '" <<
							c->parent()->path() << "." <<
							f->name() << "' not known\n";
					}
				}
			}
		}
		*/

		RecvFunc recvFunc() const {
	//		if ( lookup_ )
				return reinterpret_cast< RecvFunc >( relayFunc );
	//		else
	//			return reinterpret_cast< RecvFunc >( objLookupFunc );
		}

		const Ftype* ftype() const  {
			static const Ftype1< T > myFtype_;
			return &myFtype_;
		}

	private:
};

template < class T1, class T2 > class RelayFinfo2: public RelayFinfo
{
	public:
		RelayFinfo2( Field& f )
		:	RelayFinfo( f )
		{
			;
		}

		// The call first gets sent down to the recvFunc of the
		// original field. Then it comes back up, doing
		// all the outgoing calls (if any) from this RelayFinfo.
		static void relayFunc( Conn* c, T1 v1, T2 v2 ) {
			RelayConn* cr = dynamic_cast< RelayConn* >( c );
			if (cr) {
				RelayFinfo2< T1, T2 >* f = 
					dynamic_cast< RelayFinfo2< T1, T2 > *>( cr->finfo() );
				if ( f ) {
					// Do the original recvfunc
					RecvFunc oldfunc = f->innerFinfo()->recvFunc();
					reinterpret_cast< void ( * )( Conn*, T1, T2 ) >(oldfunc)
						( c, v1, v2 );
					// Then do the outgoing stuff if any
					for (unsigned long i = 0;
						i < f->outConn( c->parent() )->nTargets(); i++){
						reinterpret_cast< void ( * )( Conn*, T1, T2 ) >(
							f->rfunc_[ i ])( 
							f->outConn( c->parent() )->target( i ),
								v1, v2
						);
					}
				}
			}
		}

		RecvFunc recvFunc() const {
			return reinterpret_cast< RecvFunc >( relayFunc );
		}

		const Ftype* ftype() const  {
			static const Ftype2< T1, T2 > myFtype_;
			return &myFtype_;
		}
};

template < class T1, class T2, class T3 > class RelayFinfo3:
	public RelayFinfo
{
	public:
		RelayFinfo3( Field& f )
		:	RelayFinfo( f )
		{
			;
		}

		// The call first gets sent down to the recvFunc of the
		// original field. Then it comes back up, doing
		// all the outgoing calls (if any) from this RelayFinfo.
		static void relayFunc( Conn* c, T1 v1, T2 v2, T3 v3 ) {
			RelayConn* cr = dynamic_cast< RelayConn* >( c );
			if (cr) {
				RelayFinfo3< T1, T2, T3 >* f = 
					dynamic_cast< RelayFinfo3< T1, T2, T3 > *>( cr->finfo() );
				if ( f ) {
					// Do the original recvfunc
					RecvFunc oldfunc = f->innerFinfo()->recvFunc();
					reinterpret_cast< void ( * )( Conn*, T1, T2, T3 ) >(oldfunc)
						( c, v1, v2, v3 );
					// Then do the outgoing stuff if any
					for (unsigned long i = 0;
						i < f->outConn( c->parent() )->nTargets(); i++){
						reinterpret_cast< void ( * )( Conn*, T1, T2, T3 ) >(
							f->rfunc_[ i ])( 
							f->outConn( c->parent() )->target( i ),
								v1, v2, v3
						);
					}
				}
			}
		}

		RecvFunc recvFunc() const {
			return reinterpret_cast< RecvFunc >( relayFunc );
		}

		const Ftype* ftype() const  {
			static const Ftype3< T1, T2, T3 > myFtype_;
			return &myFtype_;
		}
};

// This handles inputs of zero args, and outputs of
// the specified type to value fields. The input
// is a trigger for sending the value of the field out on the
// outgoing message. It differs from other kinds of relays in that
// a trigger on this relay does NOT propagate to any other msg,
// and instead just does a field retrieval for the one outgoing msg.
template < class T > class ValueRelayFinfo: public RelayFinfo1< T >
{
	public:
		ValueRelayFinfo( Field& f )
		: RelayFinfo1< T >( f ) 
		{
			;
		}

		static void relayFunc( Conn* c ) {
			RelayConn* cr = dynamic_cast< RelayConn* >( c );
			if ( !cr ) {
				cerr << "Error: ValueRelayFinfo::relayFunc:: Failed to cast RelayConn\n";
				return;
			}

			ValueRelayFinfo< T >* f = 
				dynamic_cast< ValueRelayFinfo< T >* >( cr->finfo());
			if ( !f ) {
				cerr << "Error: ValueRelayFinfo::relayFunc:: Failed to get ValueRelayFinfo\n";
				return;
			}

			ValueFinfoBase< T >* valueFinfo = 
				dynamic_cast< ValueFinfoBase< T >* >(
					f->innerFinfo() );
			if ( !valueFinfo ) {
				cerr << "Error: ValueRelayFinfo::relayFunc:: Failed to get ValueFinfo\n";
				return;
			}

			T v = valueFinfo->value( c->parent() );
			Conn* oc = f->outConn( c->parent() );
			for (unsigned long i = 0; i < oc->nTargets(); i++)
				reinterpret_cast< void ( * )( Conn*, T ) >(
					f->rfunc_[ i ])( oc->target( i ), v);
			if ( oc->nTargets() == 0 && f->rfunc_.size() == 1 ) {
				// shared Finfo. Send return func along inConn.
				oc = f->inConn( c->parent() );
				reinterpret_cast< void ( * )( Conn*, T ) >(
					f->rfunc_[ 0 ])( oc->target( 0 ), v);
			}
		}

		RecvFunc recvFunc() const {
			return relayFunc;
		}

		// Use the default 'add' function.

		Finfo* respondToAdd( Element* e, const Finfo* sender ) {
			return 0;
		}

		bool dropRelay( unsigned long index, Conn* src ) {
			delete this;
			return 1;
		}

		// Remain invisible by  returning the null field.
		// So the ValueRelayFinfo avoids handling anything other than
		// its original value request.
		Field match( const string& s ) {
			return Field();
		}


	private:
		RecvFunc sendFunc_;
};

template < class T > Finfo* appendRelay( Finfo* f, Element* e )
{
	Field temp ( f, e );
	T* rf = new T( temp );
	if ( rf )
		temp.appendRelay( rf );
	return rf;
}

#endif // _RELAY_FINFO_H
