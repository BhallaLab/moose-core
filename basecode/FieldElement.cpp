/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2013 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"

FieldElement::FieldElement( Id parent, Id self, 
		const Cinfo* c, const string& name, 
		const FieldElementFinfoBase* fef
	)
	:
		Element( self, c, name, parent.element()->numData(), 0 ),
		parent_( parent ),
		fef_( fef )
{;}

/**
 * Here we only clear out the messages, not the data.
 * It is the responsibility of the parent Element and
 * object to clear all the data.
 * Otherwise this function is identical to the destructor for Elements.
 */
FieldElement::~FieldElement()
{
	clearCinfoAndMsgs();
}

/////////////////////////////////////////////////////////////////////////
// Element info functions
/////////////////////////////////////////////////////////////////////////

unsigned int FieldElement::numData() const
{
	return parent_.element()->numData();
}

unsigned int FieldElement::numField( unsigned int rawIndex ) const 
{
	const char* data = parent_.element()->data( rawIndex );
	assert( data );
	return fef_->getNumField( data );
}

char* FieldElement::data( unsigned int rawIndex, unsigned int fieldIndex ) 
		const
{
	char* data = parent_.element()->data( rawIndex );
	return fef_->lookupField( data, fieldIndex );
}

void FieldElement::resize( unsigned int newNumData )
{
	assert( 0 );
}

void FieldElement::resizeField( 
		unsigned int rawIndex, unsigned int newNumField )
{
	char* data = parent_.element()->data( rawIndex );
	fef_->setNumField( data, newNumField );
}
