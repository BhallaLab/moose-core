/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2009 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"

///////////////////////////////////////////////////////////////////////////
Msg::Msg( Element* src, Element* dest )
	: src_( src ), dest_( dest )
{
	;
}

Msg::~Msg()
{
	;
}

///////////////////////////////////////////////////////////////////////////

SparseMsg::SparseMsg( Element* src, Element* dest )
	: Msg( src, dest )
{
	;
}

void SparseMsg::pushQ( unsigned int srcElementIndex, double time ) const
{
	unsigned int* synIndex;
	unsigned int* elementIndex;
	unsigned int n = m_.getRow( i, &synIndex, &elementIndex );
	for ( unsigned int i = 0; i < n; ++i )
		dest_->pushQ( *elementIndex++, *synIndex++, time );
}

///////////////////////////////////////////////////////////////////////////
class One2OneMsg: publicMsg
{
	public:
		SparseMsg( Element* src, Element* dest );
		void pushQ( unsigned int srcElementIndex, double time ) const {
			dest->pushQ( srcElementIndex, synIndex, time );
		}
	private:
		unsigned int synIndex_;
};

One2OneMsg::One2OneMsg( Element* src, Element* dest )
	: Msg( src, dest )
{
	;
}

void One2OneMsg::pushQ( unsigned int srcElementIndex, double time ) const
{
	dest->pushQ( srcElementIndex, synIndex, time );
}
