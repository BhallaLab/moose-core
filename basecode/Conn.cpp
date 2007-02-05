/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2005 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU General Public License version 2
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"

Element* Conn::sourceElement() const
{
/// A bit roundabout: Looks up the target Conn to find its own Element*.
	return e_->lookupConn( index_ )->targetElement();
}

/*
Conn* Conn::targetConn()
{
	return e_->connPtr( index_ );
}
*/

unsigned int Conn::sourceIndex( const Element* e ) const
{
	return e->connIndex( this );
}

/**
 * Strictly speaking we need not pass in the index j, as we can get
 * it correctly as in the assert statement below. But it is much
 * faster to do this update from an external counter.
 */
void Conn::updateIndex( unsigned int j )
{
	assert( sourceIndex( sourceElement() ) == j );
	e_->lookupVariableConn( index_ )->index_ = j;
}

void Conn::set( Element* e, unsigned int index )
{
	e_ = e;
	index_ = index;
}
