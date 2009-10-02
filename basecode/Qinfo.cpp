/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2009 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
#include "Qinfo.h"

void Qinfo::addToQ( vector< char >& q, const char* arg ) const
{
	unsigned int origSize = q.size();
	q.resize( origSize + sizeof( Qinfo ) + size_ );
	char* pos = &( q[origSize] );
	memcpy( pos, this, sizeof( Qinfo ) );
	memcpy( pos + sizeof( Qinfo ), arg, size_ );
}

