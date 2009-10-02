/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2009 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"

Qinfo::Qinfo( FuncId f, unsigned int srcIndex, unsigned int size )
	:	m_( 0 ), 
		useSendTo_( 0 ), 
		f_( f ), 
		srcIndex_( srcIndex ),
		size_( size )
{;}

Qinfo::Qinfo( FuncId f, unsigned int srcIndex, 
	unsigned int size, bool useSendTo )
	:	m_( 0 ), 
		useSendTo_( useSendTo ), 
		f_( f ), 
		srcIndex_( srcIndex ),
		size_( size )
{;}


void Qinfo::addToQ( vector< char >& q, const char* arg ) const
{
	unsigned int origSize = q.size();
	q.resize( origSize + sizeof( Qinfo ) + size_ );
	char* pos = &( q[origSize] );
	memcpy( pos, this, sizeof( Qinfo ) );
	memcpy( pos + sizeof( Qinfo ), arg, size_ );
}

void Qinfo::expandSize()
{
	size_ += sizeof( unsigned int );
}
