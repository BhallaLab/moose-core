/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2005 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU General Public License version 2
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
#include "ValueFinfo.h"
#include "DynamicFinfo.h"

/**
* This operation requires the formation of a dynamic
* Finfo to handle the messaging, as Value fields are
* not assigned a message src or dest. We assume that the
* operation is not going to fail, so we create a DynamicFinfo.
* If it doesn't work that will sit around unused.
*/
bool ValueFinfo::add( 
	Element* e, Element* destElm, const Finfo* destFinfo
	) const 
{
	// If this is going to be a message source, then we will
	// be sending the local value through getFunc, and triggering
	// the response through trigfunc.
	DynamicFinfo *df = new DynamicFinfo( name(), this,
					set_, get_,
					set_, ftype()->trigFunc() );
	e->addFinfo( df );
	return df->add( e, destElm, destFinfo );
}
			
/**
 * There are five possible situations for 'add' requests.
 * 1. set. Here the ValueFinfo is a MsgDest, no other messages involved.
 * - SrcType is the same as self
 * - srcFl is empty
 * - returnFl is empty and has to be filled with the 'set' recvFunc.
 * - Dynamic Finfo will be used only as a MsgDest.
 * 2. trigger. Here the ValueFinfo at some point is going to be asked
 * 	to return a value.
 * - SrcType is Ftype0.
 * - srcFl is empty
 * - returnFl is empty and has to be filled with trigFunc.
 * - DynamicFinfo will be used at once as a MsgDest, and later will
 *   be asked to send messages elsewhere.
 * 3. Source. Here the ValueFinfo is being asked to return a value, but
 * there is no trigger mechanism in place as yet. This is handled by
 * the 'add' operation, not the respondToAdd.
 *
 * 4. Shared, as destination. Here the ValueFinfo is being asked to
 * be a target for a trigger, and source for a response, all at once,
 * but the message request is being made elsewhere.
 * - SrcType is Ftype0
 * - srcFl has a recvFunc<T> for the response
 * - returnFl is empty and has to be filled with trigFunc.
 * - DynamicFinfo will be set up at once both as MsgDest and MsgSrc,
 *   and needs to be supplied with the assigned recvFunc<T>.
 *
 * 5. Shared, as source. Here the ValueFinfo is being asked to connect
 * to a target to send it data, and the target handles shared
 * messages and will send back a trigger for when it wants the data.
 * This is handled by the 'add' operation, not the respondToAdd.
 *
 * Here we need to fill in the
 * location of these forthcoming src and dest arrays
 * into the new DynamicFinfo
 * before they actually exist. This is magically done by the
 * Element::addFinfo function, which also puts in
 * the location of the yet-to-be-formed Conn entry
 * because this is used to look up the DynamicFinfo.
 *
 * \todo: We could probably replace all of this by first creating the
 * DynamicFinfo and then letting it handle all the cases.
 */
bool ValueFinfo::respondToAdd(
		Element* e, Element* src, const Ftype *srcType,
		FuncList& srcFl, FuncList& returnFl,
		unsigned int& destIndex, unsigned int& numDest
) const
{
	assert( srcType != 0 );
	assert( src != 0 && e != 0 );
	assert( returnFl.size() == 0 );
	DynamicFinfo *df = new DynamicFinfo( name(), this,
					set_, get_,
					set_, ftype()->trigFunc() );
	e->addFinfo( df );
	return df->respondToAdd( e, src, srcType, srcFl, returnFl,
					destIndex, numDest );
}

/// Dummy function: DynamicFinfo should handle
void ValueFinfo::dropAll( Element* e ) const
{
		assert( 0 );
}

/// Dummy function: DynamicFinfo should handle
bool ValueFinfo::drop( Element* e, unsigned int i ) const
{
		assert( 0 );
		return 0;
}
