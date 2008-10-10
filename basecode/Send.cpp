/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2006 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
#include <algorithm>
#include "SimpleElement.h"
#include "Send.h"

void send0( Eref e, Slot src )
{
	const Msg* m = e.e->msg( src.msg() );
	if ( m->funcId() == 0 ) return;
	do {
		RecvFunc rf = m->func( src.func() );
		vector< ConnTainer* >::const_iterator i;
		// Going through the MsgSrc vector of ConnTainers
		for ( i = m->begin( ); i != m->end( ); i++ ) {
			Conn* j = ( *i )->conn( e, src.func() ); 
			for ( ; j->good(); j->increment() )
				rf( j );
			delete j;
		}
	} while ( ( m = m->next( e.e ) ) ); // An assignment, not a comparison.
}

void sendTo0( Eref e, Slot src, unsigned int tgt )
{
	// This will traverse through next() if needed, to get to the msg.
	const Msg* m = e.e->msg( src.msg() ); 
	Conn* j = m->findConn( e, tgt, src.func() );
	if ( j ) {
		RecvFunc rf = m->func( src.func() );
		rf( j );
		delete j;
	}
}

void sendBack0( const Conn* c, Slot src )
{
	const Msg* m = c->target().e->msg( src.msg() );
	RecvFunc rf = m->func( src.func() );
	const Conn* flip = c->flip( src.func() ); /// \todo Could be optimized.
	rf( flip );
	delete flip;
}
