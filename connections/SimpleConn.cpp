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
#include "SetConn.h"
#include "One2OneMapConn.h"
#include "All2OneConn.h"

SimpleConnTainer::SimpleConnTainer( Element* e1, Element* e2, 
			int msg1, int msg2,
			unsigned int eI1, unsigned int eI2,
			unsigned int i1, unsigned int i2 )
			: 
	ConnTainer( e1, e2, msg1, msg2 ),
		eI1_( eI1 ), eI2_( eI2 ),
		i1_( i1 ), i2_( i2 )
{;}

SimpleConnTainer::SimpleConnTainer( Eref e1, Eref e2, 
			int msg1, int msg2,
			unsigned int i1, unsigned int i2 )
			: 
	ConnTainer( e1.e, e2.e, msg1, msg2 ),
		eI1_( e1.i ), eI2_( e2.i ),
		i1_( i1 ), i2_( i2 )
{;}

Conn* SimpleConnTainer::conn( Eref e, unsigned int funcIndex ) const
{
	if ( e.e == e1() && e.i == eI1_ )
		return new SimpleConn( funcIndex, this );
	else if ( e.e == e2() && e.i == eI2_ )
		return new ReverseSimpleConn( funcIndex, this );

	// Failure option.
	// SetConn always has good() == 0. This makes it a safe thing
	// to return for any iterators.
	return new SetConn( Element::root(), 0 );
}

/**
 * Creates a duplicate ConnTainer for message(s) between 
 * new elements e1 and e2,
 * It checks the original version for which msgs to put the new one on.
 * e1 must be the new source element.
 * Returns the new ConnTainer on success, otherwise 0.
 */
ConnTainer* SimpleConnTainer::copy( Element* e1, Element* e2, bool isArray ) const
{
	// assert( e1->numMsg() > msg1() );
	// assert( e2->numMsg() > msg2() );
	if (isArray) {
		if ( e2->isGlobal() ) {
			return new All2OneConnTainer( e1, e2, msg1(), msg2() );
		} 
		else
			return new One2OneMapConnTainer( e1, e2, msg1(), msg2() );
	}
	else 
		return new SimpleConnTainer( e1, e2, msg1(), msg2() );
}

//////////////////////////////////////////////////////////////////////
//  SimpleConn
//////////////////////////////////////////////////////////////////////

const Conn* SimpleConn::flip( unsigned int funcIndex ) const
{
	return new ReverseSimpleConn( funcIndex, s_ );
}
