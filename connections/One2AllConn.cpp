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
#include "One2AllConn.h"

One2AllConnTainer::One2AllConnTainer( Eref e1, Eref e2, 
			int msg1, int msg2,
			unsigned int i1, unsigned int i2 )
			: 
	ConnTainer( e1.e, e2.e, msg1, msg2 ),
		eI1_( e1.i ), e2numEntries_( e2.e->numEntries() ),
		i1_( i1 ), i2_( i2 )
{;}

Conn* One2AllConnTainer::conn( Eref e, unsigned int funcIndex ) const
{
	//	numIter_++; // For reference counting. Do we need it?
	if ( e.e == e1() )
		return new One2AllConn( funcIndex, this, 0 );
	else
		return new ReverseOne2AllConn( funcIndex, this, e.i );
}

/**
 * Creates a duplicate ConnTainer for message(s) between 
 * new elements e1 and e2,
 * It checks the original version for which msgs to put the new one on.
 * e1 must be the new source element.
 * Returns the new ConnTainer on success, otherwise 0.
 */
//TODO array copy 
ConnTainer* One2AllConnTainer::copy( Element* e1, Element* e2, bool isArray ) const
{
	// assert( e1->numMsg() > msg1() );
	// assert( e2->numMsg() > msg2() );

	return new One2AllConnTainer( e1, e2, msg1(), msg2() );
}

//////////////////////////////////////////////////////////////////////
//  One2AllConn
//////////////////////////////////////////////////////////////////////

const Conn* One2AllConn::flip( unsigned int funcIndex ) const
{
	return new ReverseOne2AllConn( funcIndex, s_, index_ );
}
