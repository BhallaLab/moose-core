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


// The asyncQ on the target serves all indices within the Element.
// tgtRange: type[all,single,multiple,1range,multipleranges]: 
// 	all: All indices
// 	single: < index >
// 	multiple: < n, index, index, ... >,
// 	singleRange< index, index >
// 	multiRange< n, <index, index >, <index, index>, ...>
// Best to have the Element evaluate it, to save space in the buffer
// and as it is easier for the Element to decide how to iterate.
// Or, for all cases:
// 	<start, end>... while end < max.
// 	or, vector of < start, end >. Ugh.

// Fdata has funcId followed by data.
Msg::asend( Fdata fData )
{
	dest_->addToQ( fData, Range( 0, 1 ) );
}

///////////////////////////////////////////////////////////////////////////

SparseMsg::SparseMsg( Element* src, Element* dest )
	: Msg( src, dest )
{
	;
}

void SparseMsg::addSpike( unsigned int srcElementIndex, double time ) const
{
	const unsigned int* synIndex;
	const unsigned int* elementIndex;
	unsigned int n = m_.getRow( srcElementIndex, &synIndex, &elementIndex );
	for ( unsigned int i = 0; i < n; ++i )
		dest_->addSpike( *elementIndex++, *synIndex++, time );
}

///////////////////////////////////////////////////////////////////////////

One2OneMsg::One2OneMsg( Element* src, Element* dest )
	: Msg( src, dest ), synIndex_( 0 )
{
	;
}

void One2OneMsg::addSpike( unsigned int srcElementIndex, double time ) const
{
	dest_->addSpike( srcElementIndex, synIndex_, time );
}



