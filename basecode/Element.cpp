/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2006 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
#include "OffNodeElement.h"

const unsigned int BAD_ID = ~0;
const unsigned int MIN_NODE = 1;
const unsigned int MAX_NODE = 65536; // Dream on.

/**
 * The normal base element constructor creates an id for the element
 * as soon as it is made, and puts the element onto the elementList.
 */
Element::Element()
{
	id_ = elementList().size();
	IdLookup id;
	id.element = this;
	elementList().push_back( id );
}

/**
 * This variant is used for making dummy elements without messing
 * with the elementList. The id is hardcoded to zero here, so that
 * the destructor does not attempt to clear the location in the
 * ElementList. Note that the argument is just ignored.
 */
Element::Element( bool ignoreId )
{
	id_ = 0;
}

/**
 * The virtual destructor for Elements cleans up the entry on the
 * elementList. The special case of zero id (from above) is not
 * deleted.
 */
Element::~Element()
{
	if ( id_ != 0 )
		elementList()[ id_ ].element = 0;
}

/**
 * Here we do nasty stuff to identify off-node elements and set up
 * the appropriate OffNodeElement data structure for it to use.
 * It is a static function.
 */
Element* Element::element( unsigned int id )
{
	if ( id < elementList().size() ) {
		unsigned int node = elementList()[ id ].node;
		// Check if it is a good node
		if ( node >= MIN_NODE && node <= MAX_NODE )
			return new OffNodeElement( id, node - 1 );
		else 
			return elementList()[ id ].element;
	}
	return 0;
}

/**
 * Returns the most recently created element.
 * It is a static function.
 */
Element* Element::lastElement() {
	assert ( elementList().size() > 0 );
	return Element::element( elementList().size() - 1 );
}

vector< IdLookup >& Element::elementList()
{
	static vector< IdLookup > elementList;

	return elementList;
}

unsigned int Element::numElements()
{
	return elementList().size();
}
