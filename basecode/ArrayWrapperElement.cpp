/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "moose.h"
#include "ArrayWrapperElement.h"

ArrayWrapperElement::ArrayWrapperElement( Element* arrayElement, unsigned int index )
	: SimpleElement( Id(), "wrapper", 0, 0, 0 ),
	arrayElement_( arrayElement ), index_( index )
{
	assert( index < arrayElement->numEntries() );
}


ArrayWrapperElement::~ArrayWrapperElement( )
{
	;
}

void* ArrayWrapperElement::data() const
{
	return static_cast< void* >(
		static_cast< char* >( arrayElement_->data( ) )
		+ index_ * arrayElement_->cinfo()->ftype()->size() );
}

unsigned int ArrayWrapperElement::numEntries( ) const
{
	return arrayElement_->numEntries();
}
