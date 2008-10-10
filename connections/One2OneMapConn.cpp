/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2008 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
#include "SimpleConn.h"
#include "One2OneMapConn.h"

One2OneMapConnTainer::One2OneMapConnTainer( Eref e1, Eref e2, 
			int msg1, int msg2,
			unsigned int i1 )
			: 
	ConnTainer( e1.e, e2.e, msg1, msg2 ),
		i1_( i1 ), i2_( e2.e->numEntries() )
{
	for ( unsigned int i = 0; i < i2_.size(); i++ ) {
		i2_[i] = e2->numTargets( msg2, i );
	}
}

Conn* One2OneMapConnTainer::conn( Eref e, unsigned int funcIndex ) const
{
	//	numIter_++; // For reference counting. Do we need it?
	if ( e.e == e1() )
		return new One2OneMapConn( funcIndex, this, e.i );
	else
		return new ReverseOne2OneMapConn( funcIndex, this, e.i );
}

/**
 * Creates a duplicate ConnTainer for message(s) between 
 * new elements e1 and e2,
 * It checks the original version for which msgs to put the new one on.
 * e1 must be the new source element.
 * Returns the new ConnTainer on success, otherwise 0.
 */
 // TODO array copy
ConnTainer* One2OneMapConnTainer::copy( Element* e1, Element* e2, bool isArray ) const
{
	// assert( e1->numMsg() > msg1() );
	// assert( e2->numMsg() > msg2() );

	return new One2OneMapConnTainer( e1, e2, msg1(), msg2() );
}

//////////////////////////////////////////////////////////////////////
//  One2OneMapConn
//////////////////////////////////////////////////////////////////////

const Conn* One2OneMapConn::flip( unsigned int funcIndex ) const
{
	return new ReverseOne2OneMapConn( funcIndex, s_, index_ );
}
