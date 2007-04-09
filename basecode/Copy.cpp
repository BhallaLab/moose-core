/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "moose.h"
#include "ThisFinfo.h"
#include "../element/Neutral.h"

/*
 * This file handles copy operations for the SimpleElement class.
 */

/**
 * Virtual function. True if current element descends from the
 * specified ancestor. Uses recursion because the class of 
 * each ancestor is not known.
 * While the main use of this function is to avoid loops in element
 * trees, in principle one could connect up a loop that would
 * confuse this function too.
 */
bool SimpleElement::isDescendant( const Element* ancestor ) const
{
	if ( this == Element::root() )
			return 0;
	assert( connDestBegin( 0 ) != connDestEnd( 0 ) );
	return connDestBegin( 0 )->targetElement()->isDescendant( ancestor);
}

/**
 * This function copies the element, its data and its dynamic Finfos.
 * What it does not do is to replace any pointers to other elements
 * in the Conn array. 
 * It does not do anything about the element hierarchy either because
 * that is also handled through messages, ie., the Conn array.
 * The returned Element is dangling in memory: No parent or child.
 */

Element* SimpleElement::innerCopy() const
{
	SimpleElement* ret = new SimpleElement( *this );
	
	assert( finfo_.size() > 0 );
	assert( dynamic_cast< ThisFinfo* >( finfo_[0] ) != 0 );

	// Copy the data
	ret->data_ = finfo_[0]->ftype()->copy( data_, 1 );
	// Copy the dynamic Finfos.
	for ( unsigned int i = 1; i < finfo_.size(); i++ ) {
		ret->finfo_[i] = finfo_[i]->copy();
	}
	return ret;
}

/**
 * This function fills up the map with current element and
 * all its descendants. Returns the root element of the copied tree.
 * The first entry in the map is the original
 * The second entry in the map is the copy.
 * The function does NOT fix up the messages.
 */
Element* SimpleElement::innerDeepCopy(
	map< const Element*, Element* >& tree ) const
{
	Element* duplicate = innerCopy();
	tree[ this ] = duplicate;
	
	// The 0 slot in the MsgSrc array is for child elements.
	vector< Conn >::const_iterator i;
	vector< Conn >::const_iterator begin = connSrcBegin( 0 );
	vector< Conn >::const_iterator end = connSrcEnd( 0 );
	for ( i = begin; i != end; i++ ) {
		// Watch out for loops.
		if ( tree.find( i->targetElement() ) != tree.end() )
			cout << "Warning: SimpleElement::innerDeepCopy: Loop in element tree at " << i->targetElement()->name() << endl;
		else 
			i->targetElement()->innerDeepCopy( tree );
	}
	return duplicate;
}

/**
 * This function does a deep copy of the current element
 * including all messages. Returns the base of the copied tree.
 * It attaches the copied element tree to the parent.
 * It is non-recursive but calls lots recursive functions.
 */

Element* SimpleElement::copy( Element* parent ) const
{
	if ( parent->isDescendant( this ) ) {
		cout << "Warning: SimpleElement::copy: Attempt to copy within descendant tree" << parent->name() << endl;
		return 0;
	}
	map< const Element*, Element* > tree;
	map< const Element*, Element* >::iterator i;

	Element* child = innerDeepCopy( tree );

	for ( i = tree.begin(); i != tree.end(); i++ ) {
		i->second->replaceCopyPointers( tree );
	}
	
	// Finally, stick the copied tree onto the parent Element.
	// Here we use rather base-level calls to insert a new Conn on
	// the parent, but reuse the 'parent' Conn from the new child
	// which is the base of the copied tree.
	vector< RecvFunc >rf;
	rf.push_back( RFCAST( &Neutral::childFunc ) );
	unsigned int parentConn = parent->insertConnOnSrc( 0, rf, 0, 0 );

	// Note we are a little naughty here: using the dest_ of the
	// original element. Should be fine because the copy is, well,
	// a copy.
	unsigned int childConn = dest_[ 0 ].begin();
	parent->connect( parentConn, child, childConn );

	return child;
}

/**
 * This function replaces Element* pointers in the conn_ vector
 * with corresponding ones from the copied tree.
 * 
 * If there are any Conns going outside the tree, they have to be 
 * removed here.
 *
 * In the case of the HaloCopy (yet to be implemented ) we would
 * instead create new exterior conns to duplicate the linkages
 * of the old ones.
 */
void SimpleElement::replaceCopyPointers(
	map< const Element*, Element* >& tree )
{
	if ( conn_.size() == 0 ) return;
	map< const Element*, Element* >::iterator j;

	// Here we do a reverse iteration because we will want to delete
	// conns that are outside the tree. The delete operation munges
	// all later Conn indices, so a forward iteration would not work.
	
	for ( unsigned int i = conn_.size(); i > 0; i-- ) {
		j = tree.find( conn_[ i - 1 ].targetElement() );
		if ( j != tree.end() ) { // Inside the tree. Just replace ptrs.
			conn_[ i - 1 ].replaceElement( j->second );
		} else {
			// Outside the tree. Delete the conn and update
			// all sorts of stuff in MsgSrc and MsgDest.
			deleteHalfConn( i - 1 );
		}
	}
}
