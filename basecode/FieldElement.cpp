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
		char* ( *lookupField )( char*, unsigned int ),
		void( *setNumField )( char*, unsigned int num ),
		unsigned int ( *getNumField )( const char* ) const
	)
	:
		Element( self, c, name, 
			parent.element()->numData(), parent.element()->isGlobal() ),
		parent_( parent ),
		lookupField_( lookupField ),
		setNumField( setNumField ),
		getNumField( getNumField )
{
	c->postCreationFunc( id, this );
}

/**
 * Here we only clear out the messages, not the data.
 * It is the responsibility of the parent Element and
 * object to clear all the data.
 */
FieldElement::~FieldElement()
{
	cinfo_ = 0; // A flag that the Element is doomed, used to avoid lookups when deleting Msgs.
	for ( vector< vector< MsgFuncBinding > >::iterator i = msgBinding_.begin(); i != msgBinding_.end(); ++i ) {
		for ( vector< MsgFuncBinding >::iterator j = i->begin(); j != i->end(); ++j ) {
			// This call internally protects against double deletion.
			Msg::deleteMsg( j->mid );
		}
	}

	for ( vector< MsgId >::iterator i = m_.begin(); i != m_.end(); ++i )
		if ( *i ) // Dropped Msgs set this pointer to zero, so skip them.
			Msg::deleteMsg( *i );
}

/////////////////////////////////////////////////////////////////////////
// Element info functions
/////////////////////////////////////////////////////////////////////////

unsigned int FieldElement::numData() const
{
	return parent.element()->numData();
}

unsigned int FieldElement::numField( unsigned int rawIndex ) const 
{
	const char* data = parent.element()->data( rawIndex );
	assert( data );
	return getNumField_( data );
}

char* FieldElement::data( unsigned int rawIndex, unsigned int fieldIndex ) 
		const
{
	char* data = parent.element()->data( rawIndex );
	return lookupField_( data, fieldIndex );
}

void Field Element::resize( unsigned int newNumData )
{
	assert( 0 );
}

void Field Element::resizeField( 
		unsigned int rawIndex, unsigned int newNumField )
{
	char* data = parent.element()->data( rawIndex );
	setNumField_( data, newNumField );
}

