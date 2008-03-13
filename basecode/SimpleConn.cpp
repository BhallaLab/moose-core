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

SimpleConnTainer::SimpleConnTainer( Element* e1, Element* e2, 
			int msg1, int msg2,
			unsigned int eI1, unsigned int eI2,
			unsigned int i1, unsigned int i2 )
			: 
	ConnTainer( e1, e2, msg1, msg2 ),
		eI1_( eI1 ), eI2_( eI2 ),
		i1_( i1 ), i2_( i2 )
{;}

Conn* SimpleConnTainer::conn( unsigned int eIndex, bool isReverse ) const
{
	//	numIter_++; // For reference counting. Do we need it?
	if ( isReverse )
		return new ReverseSimpleConn( this );
	else
		return new SimpleConn( this );
}

Conn* SimpleConnTainer::conn( unsigned int eIndex, bool isReverse,
	unsigned int connIndex ) const
{
	//	numIter_++; // For reference counting. Do we need it?
	if ( connIndex != 0 )
		return 0;

	if ( isReverse )
		return new ReverseSimpleConn( this );
	else
		return new SimpleConn( this );
}

/**
 * Creates a duplicate ConnTainer for message(s) between 
 * new elements e1 and e2,
 * It checks the original version for which msgs to put the new one on.
 * e1 must be the new source element.
 * Returns the new ConnTainer on success, otherwise 0.
 */
ConnTainer* SimpleConnTainer::copy( Element* e1, Element* e2 ) const
{
	// assert( e1->numMsg() > msg1() );
	// assert( e2->numMsg() > msg2() );

	return new SimpleConnTainer( e1, e2, msg1(), msg2() );
}

//////////////////////////////////////////////////////////////////////
//  SimpleConn
//////////////////////////////////////////////////////////////////////

const Conn* SimpleConn::flip() const
{
	return new ReverseSimpleConn( s_ );
}
