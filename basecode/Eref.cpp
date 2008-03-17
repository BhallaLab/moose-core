/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2008 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"


void* Eref::data()
{
	return e->data( i );
}

bool Eref::operator<( const Eref& other ) const
{
	if ( e == other.e )
		return ( i < other.i );

	return ( e < other.e );
}

Id Eref::id()
{
	Id ret = e->id();
	ret.assignIndex( i );
	return ret;
}
