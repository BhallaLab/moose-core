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
#include "All2OneConn.h"

All2OneConnTainer::All2OneConnTainer( Eref e1, Eref e2, 
			int msg1, int msg2,
			unsigned int i1, unsigned int i2 )
			: 
	ConnTainer( e1.e, e2.e, msg1, msg2 ),
		eI2_( e2.i ), e1numEntries_( e1.e->numEntries() ),
		i1_( i1 ), i2_( i2 )
{;}

/**
 * Initiates an All2OneConn
 * Eref e is the originating element.
*/
Conn* All2OneConnTainer::conn( Eref e, unsigned int funcIndex ) const
{
	if ( e.e == e1() )
		return new All2OneConn( funcIndex, this, e.i );
	else
		return new ReverseAll2OneConn( funcIndex, this, e.i, e1numEntries_ );
}

/**
 * Creates a duplicate ConnTainer for message(s) between 
 * new elements e1 and e2,
 * It checks the original version for which msgs to put the new one on.
 * e1 must be the new source element.
 * Returns the new ConnTainer on success, otherwise 0.
 */
//TODO array copy 
ConnTainer* All2OneConnTainer::copy( Element* e1, Element* e2, bool isArray ) const
{
	// assert( e1->numMsg() > msg1() );
	// assert( e2->numMsg() > msg2() );

	return new All2OneConnTainer( e1, e2, msg1(), msg2() );
}

//////////////////////////////////////////////////////////////////////
//  All2OneConn
//////////////////////////////////////////////////////////////////////

const Conn* All2OneConn::flip( unsigned int funcIndex ) const
{
	return new ReverseAll2OneConn( funcIndex, s_, index_, s_->numSrc( target().i ) );
}
