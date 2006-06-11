/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2005 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU General Public License version 2
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"

void finfoErrorMsg( const string& name, Field dest )
{
	cerr << "Error: " << name << "add() : Invalid dest field '" <<
		dest->name() << "' on elm '" <<
		dest.getElement()->name() << "'\n";
}


void fillSrc( vector< Field >& list, Element* e, 
	vector< Finfo* >& f)
{
	vector < Finfo* >::iterator i;
	for ( i = f.begin(); i != f.end(); i++)
		list.push_back( Field( *i, e ) );
}

//////////////////////////////////////////////////////////////////
// SingleSrcFinfo
//////////////////////////////////////////////////////////////////

Finfo* SingleSrcFinfo::respondToAdd( Element* e, const Finfo* sender )
{
	if ( isSameType( sender ) ) {
		Finfo* ret = this->makeRelayFinfo( e );
		e->appendRelay( ret );
		return ret;
	}
	return 0;
}

bool SingleSrcFinfo::strGet( Element* e, string& val )
{
	vector< Field > list;
	src( list, e );
	if ( list.size() == 0 )
		val = "";
	else
		val = list[ 0 ].path();
	return 1;
}

bool SingleSrcFinfo::add( Element* e, Field& destfield,
	bool useSharedConn )
{
	if ( sharesConn_ ) {
		// Here we fill in the recvFunc for a shared conn
		if ( useSharedConn ) {
			cerr << "Error: SingleSrcFinfo::add: cannot add when sharing func, should use addRecvFunc\n";
			// getSrc_( e )->addRecvFunc( destfield->recvFunc(), 0 );
			return 0;
		} else { // Cannot add regular messages, do relay.
			Finfo* rf = makeRelayFinfo( e );
			if ( rf->add( e, destfield ) ) {
				e->appendRelay( rf );
				return 1;
			}
			rf->destroy();
		}
		return 0;
	}
	if ( useSharedConn ) {
		finfoErrorMsg("SingleSrcFinfo::add: useSharedConn = 1 != sharesConn",
		destfield );
		return 0;
	}

	Finfo* dest = destfield.respondToAdd( this );
	if ( !dest )
		return 0; // Target does not like me.

	if ( getSrc_( e )->add(
		dest->recvFunc(), 
		dest->inConn( destfield.getElement() )  )) 
		return 1;

	finfoErrorMsg( "SingleSrc0Finfo", destfield );
	return 0;
}


//////////////////////////////////////////////////////////////////
// SingleSrc0Finfo
//////////////////////////////////////////////////////////////////

void SingleSrc0Finfo::relayFunc( Conn* c )
{
	RelayConn* cr = dynamic_cast< RelayConn* >( c );
	if (cr) {
		SingleSrc0Finfo *f = 
			dynamic_cast< SingleSrc0Finfo *>( cr->finfo() );
		if (f)
			static_cast< SingleMsgSrc0* >( f->getSrc_( c->parent() ) )
				->send();
	}
}

/*
Finfo* SingleSrc0Finfo::respondToAdd( Element* e, const Finfo* sender )
{
	if ( isSameType( sender ) )
		return appendRelay< RelayFinfo0 >( this, e );
	return 0;
}
*/
