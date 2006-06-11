/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2005 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _SOLVE_FINFO_H
#define _SOLVE_FINFO_H

///////////////////////////////////////////////////////////////////////
// Here we define a set of Finfo classes that handle message
// to and from solvers.
///////////////////////////////////////////////////////////////////////

// Break up the solver requirements.
// First, 

class SolverFinfo: public Finfo
{
	public:
		SolverFinfo( const string& fieldList,  )
		// ValueLookup is a map of Set/Get functions for each field
		// name, that use the conn as an argument. Much like the args
		// for ValueFinfo.
		// We also need a way to set up messages. These could be spun
		// off as functor objects that encapsulate ptr info.
		// We need a way to decide which existing messages need to
		// be re-handled. Presumably anything ocurring outside the 
		// ambit of the solved classes.
	private:
};

// This is attached to the solved object.
// Need a part that is statically defined and shared
// Need a part that handles the specifics.
// This is the shared part. The local part has Conn info and not
// much else, see the piggyFinfo below.
class SolvedFinfo: public Finfo
{
	public:
		SolvedFinfo( const string& fieldList )
	private:
		
};

// Message source that piggybacks onto an existing conn and
// does dynamic lookup of target functions. Used for very
// rarely used messages where we would rather not have any
// memory overhead at all.
// May also be good for solvers.
class PiggyFinfo: public Finfo
{
	public:
		PiggyFinfo( Conn* (*getConn)( Element* ) ,
			const string& destfield )
		:	Finfo( f->name() ),	getConn_( getConn ),
		{
			;
		}

		RecvFunc targetFunc( Element* e, unsigned long i ) const {
			// Look up the target element on the Conn list.
			// Then ask it for its recvFunc.
			Element* tgt = getConn_( e )->target( i );
			if ( !tgt )
				return 0;

			return tgt->field( destfield_ )->recvFunc();
		}

		// Won't bother with this just yet.
		unsigned long matchRemoteFunc(
			Element* e, RecvFunc rf ) const {
				return 0;
		}

		void addRecvFunc( Element* e, RecvFunc rf,
			unsigned long position ) {
			;
		}

		Conn* inConn( Element* e ) const {
			return 0;
		}

		Conn* outConn( Element* e ) const {
			return getConn_( e );
		}

		// Won't bother with this just yet.
		void src( vector< Field >& list, Element* e ) {
			;
		}

		// Won't bother with this just yet.
		void dest( vector< Field >& list, Element* e ) {
			;
		}

		// This is illegal on its own. Must be shared.
		bool add( Element* e, Field& destfield, bool useSharedConn = 0)
		{
			return 0;
		}

		Finfo* respondToAdd( Element* e, const Finfo* sender ) {
			return 0;
		}

		void initialize( const Cinfo* c ) {
			;
		}

	private:
};

class PiggyFinfo0: public PiggyFinfo
{
	public:
		PiggyFinfo0( const string& name, Conn* (*getConn)( Element* ) ,
			const string& destfield )
		:	PiggyFinfo( name, getConn, destfield ) {
			;
		}

		Finfo* makeRelayFinfo( Element* e ) {
			Field temp( this, e );
			return new RelayFinfo0( temp );
		}

		void send( Element* e ) const {
			vector< Conn* > tgt;
			vector< Conn* >::iterator i;
			getConn_( e )->listTargets( tgt );
			RecvFunc func;
			for ( i = tgt.begin(); i != tgt.end(); i++ ) {
				func = ( *i )->parent()->field( destfield_)->recvFunc();
				if ( func ) {
					func( *i );
				}
			}
		}
}


class PiggyFinfo1< T >: public PiggyFinfo
{
	public:
		PiggyFinfo1( const string& name, Conn* (*getConn)( Element* ) ,
			const string& destfield )
		:	PiggyFinfo( name, getConn, destfield ) {
			;
		}

		Finfo* makeRelayFinfo( Element* e ) {
			Field temp( this, e );
			return new RelayFinfo1< T >( temp );
		}

		void send( Element* e, T v ) const {
			vector< Conn* > tgt;
			vector< Conn* >::iterator i;
			getConn_( e )->listTargets( tgt );
			RecvFunc func;
			for ( i = tgt.begin(); i != tgt.end(); i++ ) {
				func = ( *i )->parent()->field( destfield_)->recvFunc();
				if ( func ) {
					reinterpret_cast< void ( * )( Conn*, T ) > (
						func )( *i, v );
				}
			}
		}
}

#endif // _SOLVE_FINFO_H
