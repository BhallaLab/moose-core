/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2005 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _SHARED_FINFO_H
#define _SHARED_FINFO_H

///////////////////////////////////////////////////////////////////////
// Here we define a set of classes derived from Finfo0 and Finfo1
// that handle group messaging using shared conns.
// There is only one class, handling both ends of the
// group message.
///////////////////////////////////////////////////////////////////////

extern bool parseShareList( 
	const Cinfo* c, const string&,
	vector< Finfo* >&, vector< Finfo* >&
);

// This finfo handles messages that share a common Conn. A single
// 'add' call to a SharedFinfo sets up all these messages. There can
// be both incoming and outgoing messages. In order to connect to a
// SharedFinfo there must be a perfect, ordered match between the
// incoming and outgoing messages of the two SharedFinfos.
// The name is just the name of the msgsrc
// The getSrc returns a ptr to the msgsrc, given the element ptr.
// The triggers are a list of msgdests that trigger this msgsrc.
// Here we just put in their finfo names.
class SharedFinfo: public Finfo
{
	public:
		SharedFinfo( const string& name, 
			Conn* ( *getConn )( Element* ),
			const string& connSharers
		)
		:	Finfo(name), 
			getConn_( getConn ),
			connSharers_( connSharers ),
			myFtype_( finfos_ )
		{
			;
		}

		RecvFunc recvFunc() const {
			return 0;
		}

		RecvFunc targetFunc( Element* e, unsigned long i ) const {
			return 0;
		}
		
		// returns number of matches
		unsigned long matchRemoteFunc( Element* e, RecvFunc rf ) const
		{
			return 0;
		}

		void addRecvFunc( Element* e, RecvFunc rf, unsigned long position ) {
			;
		}

		bool add( Element* e, Field& destfield, bool useSharedConn = 0);

		Conn* inConn( Element* e ) const {
			return getConn_( e );
		}

		Conn* outConn( Element* e ) const {
			return getConn_( e );
		}

		void src( vector< Field >& list, Element* e ) {
			;
		}

		bool funcDrop( Element* e, const Conn* c ) {
			return 0;
		}

		void initialize( const Cinfo* c );

		Finfo* respondToAdd( Element* e, const Finfo* sender );
		
		const Ftype* ftype() const {
			return &myFtype_;
		}

		Finfo* makeRelayFinfo( Element* e ) {
			return 0;
		}

	protected:

		void setInFinfos( Finfo* f ) {
			finfos_.push_back( f );
			sharedIn_.push_back( f );
		}

		void setOutFinfos( Finfo* f ) {
			finfos_.push_back( f );
			sharedOut_.push_back( f );
		}

	private:
		Conn* ( *getConn_ )( Element * );
		string connSharers_;
		vector< Finfo* > finfos_;
		vector< Finfo* > sharedOut_;
		vector< Finfo* > sharedIn_;
		MultiFtype myFtype_;
};

/*
// This Finfo is for messages sharing a common Conn, but in a somewhat
// specialized context. This is a destination Finfo from a Shared
// Finfo. It expects that the incoming message wants an immediate
// response by a return function. Further it expects that each incoming
// message wants an independent response from all the other messages.
// For example, if there are 100 instances of an HHChannel, all using
// the same HHGate, each will independently send a message to the 
// HHGate asking it to calculate their state variables and return 
// updated state values. None of the HHChannels is interested in
// what the other channels are doing.
// In principle this kind of Finfo can handle multiple output messages,
// but for now the idea is it just returns to sender.
// There are some similarities to Synapse1Finfo

class ReturnFinfo: public SharedFinfo
{
	public:
		ReturnFinfo:( const string& name,
			vector< Conn* >& ( *getConnVec )( Element* ),
			unsigned long( *newConn )( Element* ) ,
			const string& connSharers )
		:	SharedFinfo( name, 0, connSharers ),
			getConnVec_( getConnVec ),
			newConn_( newConn )
		{
			;
		}

		Field match( const string& s )
		{
			if ( s == name() ) {
				return this;
			}
			int i = findIndex( s, name() );
			if ( i >= 0 ) {
				ReturnFinfo* ret = new ReturnFinfo( *this );
				ret->isIndex_ = 1;
				ret->index_ = i;
				return ret;
			} else {
				return Field();
			}
		}
	
		Finfo* copy() {
			if ( isIndex_ )
				return ( new ReturnFinfo( *this ) );
			else
				return this;
		}

		void destroy() const {
			if ( isIndex_ )
				delete this;
		}

		Conn* inConn( Element* e ) const {
			if ( isIndex_ ) {
				if ( getConnVec_( e ).size() > index_ )
					return getConnVec_( e )[ index_ ];
			} else {
				Conn* ret = new ReturnConn( e );
				getConnVec_( e ).push_back( ret );
				return ret;
			}
		}

		Conn* outConn( Element* e ) const {
			if ( isIndex_ ) {
				if ( getConnVec_( e ).size() > index_ )
					return getConnVec_( e )[ index_ ]->target( 0 );
			}
			return dummyConn();
		}
	
		void src( vector< Field >& list, Element* e );

		// This should only be used as a destination.
		bool add( Element* e, Field& destfield, bool useSharedConn = 0)
		{
			return 0;
		}
		Finfo* respondToAdd( Element* e, const Finfo* sender );

	private:
		vector< Conn* >& ( *getConnVec_ )( Element* );
		unsigned long( *newConn_ )( Element* );
};
*/

#endif // _SHARED_FINFO_H
