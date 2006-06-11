/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2005 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _DEST_FINFO_H
#define _DEST_FINFO_H

extern void parseTriggerList(
	const Cinfo* c, const string& triggers, vector< Finfo* >& list
);

///////////////////////////////////////////////////////////////////////
// Here we define a set of classes derived from Finfo0 and Finfo1
// that handle messaging.
// Dest0Finfo: A destination class that handles 0 fields
// Dest1Finfo<T>: A destination class that handles 1 fields
// Synapse1Finfo<T>: A synaptic destination class that handles 1 fields.
// Dest2Finfo<T1, T2>: A destination class that handles 2 fields
///////////////////////////////////////////////////////////////////////

template< class T > bool destAdd( Element* e, Field& df, Finfo* d )
{
	Finfo* f = df.respondToAdd( d );
	if ( f ) {
		Finfo* rf = appendRelay< T >( d, e );
		if ( rf )
			return rf->add( e, df );
	}
	return 0;
}

// The name is just the name of the msgdest
// rFunc is the receive function of the msgdest.
// The getConn func returns the connection given the element.
// The triggers are a list of msgsrcs that are triggered by this dest.
// Here we just put in their finfo names, later they are looked up 
// during initialization.
class DestFinfo: public Finfo
{
	public:
		DestFinfo( const string& name, 
			RecvFunc rFunc,
			Conn* ( *getConn )( Element * ),
			const string& triggers,
			bool sharesConn
		)
		:	Finfo(name), 
			recvFunc_(rFunc),
			getConn_( getConn ),
			triggers_(triggers),
			sharesConn_( sharesConn )
		{
			;
		}

		RecvFunc recvFunc() const {
			return recvFunc_;
		}

		RecvFunc targetFunc( Element* e, unsigned long i ) const;

		void addRecvFunc( Element* e, RecvFunc rf,
			unsigned long position );

		Conn* inConn( Element* e ) const {
			return getConn_( e );
		}

		Conn* outConn( Element* e ) const {
			return dummyConn();
		}

		// src is handled effectively by the default in Finfo.cpp.

		void dest( vector< Field >& list, Element* e );
		Finfo* respondToAdd( Element* e, const Finfo* sender );
		bool strGet( Element* e, string& val );

		// Returns nonzero if the remote func belongs to one of its
		// targets. Here this means that if the func has come from
		// one of the SrcFinfos triggered by this dest, return 1.
		unsigned long matchRemoteFunc( Element* e, RecvFunc func )
			const;

		void initialize( const Cinfo* c ) {
			parseTriggerList( c, triggers_, internalDest_ );
		}

	private:
		void ( *recvFunc_ )( Conn* );
		Conn* ( *getConn_ )( Element * );
		string triggers_;
		bool sharesConn_;
		vector< Finfo* > internalDest_;
};

class Dest0Finfo: public DestFinfo
{
	public:
		Dest0Finfo( const string& name, 
			void ( *rFunc )( Conn* ),
			Conn* ( *getConn )( Element * ),
			const string& triggers,
			bool sharesConn = 0
		)
		:	DestFinfo( name, rFunc, getConn, triggers, sharesConn )
		{
			;
		}

		// This adds a message that uses the arguments coming into
		// this dest and sends them elsewhere. It involves creation
		// of a relay
		bool add( Element* e, Field& destfield, bool useSharedConn = 0){
			return destAdd< RelayFinfo0 >( e, destfield, this );
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

template <class T> class Dest1Finfo: public DestFinfo
{
	public:
		Dest1Finfo( const string& name, 
			void ( *rFunc )( Conn*, T ),
			Conn* ( *getConn )( Element * ),
			const string& triggers,
			bool sharesConn = 0
		)
		:	DestFinfo( name, reinterpret_cast< RecvFunc >( rFunc ),
			getConn, triggers, sharesConn )
		{
			;
		}

		// This adds a message that uses the arguments coming into
		// this dest and sends them elsewhere. It involves creation
		// of a relay
		bool add( Element* e, Field& destfield, bool useSharedConn = 0){
			return destAdd< RelayFinfo1< T > >( e, destfield, this );
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

template <class T1, class T2> class Dest2Finfo: public DestFinfo
{
	public:
		Dest2Finfo( const string& name, 
			void ( *rFunc )( Conn*, T1, T2 ),
			Conn* ( *getConn )( Element * ),
			const string& triggers,
			bool sharesConn = 0
		)
		:	DestFinfo( name, reinterpret_cast< RecvFunc >( rFunc ),
			getConn, triggers, sharesConn )
		{
			;
		}

		// This adds a message that uses the arguments coming into
		// this dest and sends them elsewhere. It involves creation
		// of a relay
		bool add( Element* e, Field& destfield, bool useSharedConn = 0){
			return destAdd< RelayFinfo2< T1, T2 > >( e, destfield, this );
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


template <class T1, class T2, class T3> class Dest3Finfo:
	public DestFinfo
{
	public:
		Dest3Finfo( const string& name, 
			void ( *rFunc )( Conn*, T1, T2, T3 ),
			Conn* ( *getConn )( Element * ),
			const string& triggers,
			bool sharesConn = 0
		)
		:	DestFinfo( name, reinterpret_cast< RecvFunc >( rFunc ),
			getConn, triggers, sharesConn )
		{
			;
		}

		// This adds a message that uses the arguments coming into
		// this dest and sends them elsewhere. It involves creation
		// of a relay
		bool add( Element* e, Field& destfield, bool useSharedConn = 0){
			return destAdd< RelayFinfo3< T1, T2, T3 > >(
				e, destfield, this );
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

extern void synSrc(
	vector< Field >& list, vector< Conn* >& conns, 
	RecvFunc f, unsigned long index, bool isIndex
);

extern void synSrc( 
	vector< Field >& list, 
	vector< Conn* >& conns, 
	RecvFunc f,
	unsigned long index,
	bool isIndex
);

// Combines attributes of DestFinfo and ArrayFinfo
template <class T> class Synapse1Finfo: public DestFinfo
{
	public:
		Synapse1Finfo( const string& name, 
			void( *rFunc )( Conn*, T ),
			vector< Conn* >& ( *getConnVec )( Element* ),
			unsigned long ( *newConn )( Element* ),
			const string& triggers
		)
		:	DestFinfo( name, 
				reinterpret_cast< RecvFunc >( rFunc ), 0, triggers, 0 ),
			getConnVec_( getConnVec ),
			newConn_( newConn ),
			index_( 0 ),
			isIndex_( 0 )
		{
			;
		}

		// Either we refer to the entire collection of synconns, or
		// want to look up individual entries by index.
		Field match( const string& s )
		{
			if ( s == name() ) {
				return this;
			}

			int i = findIndex( s, name() );
			if ( i >= 0 ) {
				Synapse1Finfo* ret = new Synapse1Finfo( *this );
				ret->isIndex_ = 1;
				ret->index_ = i;
				return ret;
			} else {
				return Field();
			}
		}

		Finfo* copy() {
			if ( isIndex_ )
				return ( new Synapse1Finfo( *this ) );
			return this;
		}

		void destroy() const {
			if ( isIndex_ )
				delete this;
		}

		// Don't know what to do for the non-index case.
		// Fortunately most needs are handled by the traversal function
		// 'src' below.
		// For the non-index case (which is called when deleteing)
		// we need to identify which of the conns is to be removed.
		Conn* inConn( Element* e ) const {
			if ( isIndex_ ) {
				if ( getConnVec_( e ).size() > index_ )
					return getConnVec_( e )[ index_ ];
			}
			return dummyConn();
		}

		// Here we override the src function. We provide
		// either the single source or all of them, depending on the
		// field status.
		void src( vector< Field >& list, Element* e ) {
			synSrc( list, getConnVec_( e ), recvFunc(), index_, isIndex_ );
		}

		// This adds a message that uses the arguments coming into
		// this dest and sends them elsewhere. It involves creation
		// of a relay. Applies only to indexed connections, not to the
		// whole array.
		bool add( Element* e, Field& destfield, bool useSharedConn = 0){
			if ( isIndex_ ) {
				return destAdd< RelayFinfo1< T > >( e, destfield, this);
			}
			return 0;
		}

		bool drop( Element* e, Field& srcfield ) {
			Conn* oconn = srcfield->outConn( srcfield.getElement() );
			if ( isIndex_ ) {
				return oconn->disconnect( inConn( e ) );
			} else {
				vector< Conn* > c = getConnVec_( e );
				// Should really use STL find_if here.
				vector< Conn* >::iterator i;
				//for (i = c.end() - 1; i != c.begin() - 1; i-- )
				for (i = c.begin(); i != c.end(); i++)
				{
					if ( ( *i )->target( 0 ) == oconn ) {
						// Decide if we want to eliminate the entry
						// in the Conn vector here.
						return ( oconn->disconnect( *i ) != Conn::MAX );
					}
				}
			}
			return 0;
		}

		// Two cases:
		// 1. Is indexed: Return Relay Finfo to trigger an indexed conn.
		// 2. not indexed: Create a new connection
		// Here we do something peculiar. We create a new conn and
		// return an incarnation of this finfo that points to it.
		// The incarnation should have its destroy() called when done.
		// This should happen in a SrcFinfo.
		Finfo* respondToAdd( Element* e, const Finfo* sender ) {
			if ( isSameType( sender ) ) {
				if ( isIndex_ ) {
					return appendRelay< RelayFinfo1< T > >( this, e );
				} else {
					unsigned long i = newConn_( e );
					Synapse1Finfo* ret = new Synapse1Finfo( *this );
					ret->index_ = i;
					ret->isIndex_ = 1;
					return ret;
				}
			}
			return 0;
		}
	
		const Ftype* ftype() const {
			static const Ftype1< T > myFtype_;
			return &myFtype_;
		}

		Finfo* makeRelayFinfo( Element* e ) {
			Field temp( this, e );
			return new RelayFinfo1< T >( temp );
		}

	private:
		vector< Conn* > & ( *getConnVec_ )( Element * );
		unsigned long ( *newConn_ )( Element * );
		unsigned long index_;
		bool isIndex_;
};

#endif // _DEST_FINFO_H
