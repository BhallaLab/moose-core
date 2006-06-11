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

//////////////////////////////////////////////////////////////////
// NSrcFinfo
//////////////////////////////////////////////////////////////////
bool NSrcFinfo::add( Element* e, Field& destfield, bool useSharedConn )
{
	if ( sharesConn_ ) { // Conn is already done, just set the function
		if ( useSharedConn ) {
			cerr << "Error: NSrcFinfo::add: cannot add when sharing func, should use addRecvFunc\n";
			// getSrc_( e )->addRecvFunc( destfield->recvFunc() );
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
		finfoErrorMsg("NSrcFinfo::add: useSharedConn = 1 != sharesConn",
		destfield );
		return 0;
	}
	Finfo* dest = destfield.respondToAdd( this );
	if ( !dest )
		return 0; // Target does not like me.

	if ( getSrc_( e )->add(
		dest->recvFunc(), 
		dest->inConn( destfield.getElement() )  ) ) 
		return 1;

	finfoErrorMsg( "NSrc0Finfo", destfield );
	return 0;
}


Finfo* NSrcFinfo::respondToAdd( Element* e, const Finfo* sender )
{
	if ( isSameType( sender ) ) {
		Finfo* ret = this->makeRelayFinfo( e );
		e->appendRelay( ret );
		return ret;
	}
	return 0;
}

bool NSrcFinfo::strGet( Element* e, string& val )
{
	vector< Field > list;
	dest( list, e );
	val = "";
	for (unsigned int i = 0; i < list.size(); i++) {
		val += list[ i ].path();
		if (i < list.size() - 1 )
			val += ", ";
	}
	return 1;
}

//////////////////////////////////////////////////////////////////
// NSrc0Finfo
//////////////////////////////////////////////////////////////////
void NSrc0Finfo::relayFunc( Conn* c )
{
	RelayConn* cr = dynamic_cast< RelayConn* >( c );
	if (cr) {
		NSrc0Finfo *f = dynamic_cast< NSrc0Finfo* >( cr->finfo() );
		if (f)
			static_cast< NMsgSrc0* >( f->getSrc_( c->parent() ) )
				->send();
	}
}


/*
Finfo* NSrc0Finfo::respondToAdd( Element* e, const Finfo* sender )
{
	if ( isSameType( sender ) )
		return appendRelay< RelayFinfo0 >( this, e );
	return 0;
}
*/
