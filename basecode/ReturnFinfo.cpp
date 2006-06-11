/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2006 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU General Public License version 2
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"

//////////////////////////////////////////////////////////////////
// ReturnFinfo
//////////////////////////////////////////////////////////////////

// The add is not normally used, because this class is usually a
// target in a SharedFinfo context.
bool ReturnFinfo::add( Element* e, Field& destfield, bool useSharedConn)
{
	if ( sharesConn_ ) { // Conn is already done, just set the function
		if ( useSharedConn ) {
			cerr << "Error: ReturnFinfo::add: cannot add when sharing func, should use addRecvFunc\n";
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
		finfoErrorMsg("ReturnFinfo::add: useSharedConn = 1 != sharesConn",
		destfield );
		return 0;
	}
	Finfo* dest = destfield.respondToAdd( this );
	if ( !dest )
		return 0; // Target does not like me.

	if ( 
		getConn_( e )->connect( 
			dest->inConn( destfield.getElement() ),
			0, 0
			)
		) {
		getConn_( e )->addRecvFunc( dest->recvFunc() );
		return 1;
	}

	finfoErrorMsg( "Return0Finfo", destfield );
	return 0;
}


Finfo* ReturnFinfo::respondToAdd( Element* e, const Finfo* sender )
{
	if ( isSameType( sender ) ) {
		return this;
	}
	return 0;
}

bool ReturnFinfo::strGet( Element* e, string& val )
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

void ReturnFinfo::src( vector< Field >& list, Element* e )
{
	unsigned long i;
	MultiReturnConn* c = getConn_( e );
	for (i = 0 ; i < c->nTargets(); i++) {
		list.push_back( c->target( i )->parent()->lookupDestField( 
			c->target( i ), c->targetFunc( i ) ) );
	}
}
