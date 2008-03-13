/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2006 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "moose.h"
#include "IdManager.h"
#include "ThisFinfo.h"
// #include "ArrayWrapperElement.h"
#ifdef USE_MPI
const Cinfo* initPostMasterCinfo();
#endif

static Element* UNKNOWN_NODE = reinterpret_cast< Element* >( 1L );
// const unsigned int MIN_NODE = 1;
// const unsigned int MAX_NODE = 65536; // Dream on.
const unsigned int IdManager::numScratch = 1000;
const unsigned int IdManager::blockSize = 1000;
const unsigned int BAD_NODE = ~0;

IdManager::IdManager()
	: myNode_( 0 ), numNodes_( 1 ), 
	loadThresh_( 2000.0 ),
	scratchIndex_( 2 ), mainIndex_( 2 )
	// Start at 2 because root is 0 and shell is 1.
{
	elementList_.resize( blockSize );
	post_.resize( 1 );
}

void IdManager::setNodes( unsigned int myNode, unsigned int numNodes )
{
	myNode_ = myNode;
	numNodes_ = numNodes;
	elementList_[0] = Element::root();
	if ( numNodes > 1 ) {
		if ( myNode == 0 )
			nodeLoad.resize( numNodes );
		elementList_.resize( numScratch + blockSize );
		mainIndex_ = numScratch;
		post_.resize( numNodes );
	} else {
		post_.resize( 1 );
		post_[0] = 0;
	}
}

void IdManager::setPostMasters( vector< Element* >& post )
{
	unsigned int j = 0;
	for ( unsigned int i = 0; i < numNodes_; i++ ) {
		if ( i == myNode_ )
			post_[ i ] = 0;
		else
			post_[ i ] = post[ j++ ];
	}
}

/**
 * Returns the next available id and allocates space for it.
 * Later can be refined to mop up freed ids. 
 * Don't bother with the scratch space if we are on master node.
 */
unsigned int IdManager::scratchId()
{
	if ( numNodes_ <= 1 ) {
		lastId_ = mainIndex_;
		mainIndex_++;
		if ( mainIndex_ >= elementList_.size() )
			elementList_.resize( mainIndex_ * 2 );
		return lastId_;
	} else {
		if ( scratchIndex_ < numScratch ) {
			lastId_ = scratchIndex_;
			++scratchIndex_;
			return lastId_;
		} else {
			regularizeScratch();
			return ( lastId_ = scratchIndex_ );
		}
	} 
	return lastId_;
}

// Should only be called on master node, and parent should have been checked
unsigned int IdManager::childId( unsigned int parent )
{
#ifdef USE_MPI
	assert( myNode_ == 0 );
	if ( parent < mainIndex_ ) {
		Element* pa = elementList_[ parent ];
		if ( pa->cinfo() == initPostMasterCinfo() ) {
			lastId_ = mainIndex_;
			elementList_[ lastId_ ] = pa;
			// This assignment tells the system to put the object on
			// the specified node.
			mainIndex_++;
		} else { // parent is on master node.
			// Do some fancy load balancing calculation here
			unsigned int targetNode = 
				static_cast< unsigned int >( mainIndex_ / loadThresh_ ) %
				numNodes_;
			lastId_ = mainIndex_;
			elementList_[ lastId_ ] = post_[ targetNode ];
			mainIndex_++;
		}
		if ( mainIndex_ >= elementList_.size() )
			elementList_.resize( mainIndex_ * 2, 0 );
		return lastId_;
	}
	assert( 0 );
#else
	lastId_ = mainIndex_;
	mainIndex_++;
	if ( mainIndex_ >= elementList_.size() )
		elementList_.resize( mainIndex_ * 2, 0 );
	return lastId_;
#endif
}

/**
 * This variant of childId forces creation of object on specified node,
 * provided that we are in parallel mode. Otherwise it ignores the node
 * specification.  
 * Should only be called on master node, and parent should have been checked
 * In single-node mode it is equivalent to the scratchId and childId ops.
 */
unsigned int IdManager::makeIdOnNode( unsigned int childNode )
{
	lastId_ = mainIndex_;
	mainIndex_++;
#ifdef USE_MPI
	assert( myNode_ == 0 );
	assert( childNode < numNodes_ );
	if ( childNode > 0 ) { // Off-node element
		elementList_[ lastId_ ] = post_[ childNode ];
	}
#endif
	if ( mainIndex_ >= elementList_.size() )
		elementList_.resize( mainIndex_ * 2, 0 );
	return lastId_;
}

Element* IdManager::getElement( const Id& id ) const
{
	static ThisFinfo dummyFinfo( initNeutralCinfo(), 1 );
	if ( id.id_ < mainIndex_ ) {
	Element* ret = elementList_[ id.id_ ];
	if ( ret == 0 )
		return 0;
#ifdef USE_MPI
		if ( ret == UNKNOWN_NODE )
			// don't know how to handle this yet. It should trigger
			// a request to the master node to update the elist.
			// We then get into managing how many entries are unknown...
			assert( 0 );
		if ( ret->cinfo() != initPostMasterCinfo() ) {
			return ret;
		} else if ( ret->id().id_ == id.id_ ) {
			// This is the postmaster itself
			return ret;
		} else {
			OffNodeInfo* oni = new OffNodeInfo( ret, Id( index ) );

			Element* wrap = new SimpleElement( Id(), "wrapper", 0, 0, oni );
			wrap->addFinfo( &dummyFinfo );
			return wrap;
		}
#else
	if ( id.index_ > 0 ) {
		/*
		if ( ret->numEntries() > id.index_ )
			return new ArrayWrapperElement( ret , id.index_ );
		else
			return 0;
		*/
	} else {
		return ret;
	}
#endif
	}
	return 0;
}

bool IdManager::setElement( unsigned int index, Element* e )
{
	if ( index < mainIndex_ ) {
		Element* old = elementList_[ index ];
		if ( old == 0 || old == UNKNOWN_NODE ) {
			elementList_[ index ] = e;
			return 1;
		} else if ( e == 0 ) {
			// Here we are presumably clearing out an element. Permit it.
			elementList_[ index ] = 0;
			/// \todo: We could add this element to a list for reuse here.
			return 1;
		} else if ( index == 0 ) {
			// Here it is the wrapper for off-node objects. Ignore it.
			return 1;
		} else { // attempt to overwrite element. Should I assert?
			assert( 0 );
			return 1;
		}
	} else {
		if ( myNode_ > 0 ) {
			// Here we have been told by the master node to make a child 
			// at a specific index before the elementList has been
			// expanded to that index. Just expand it to fit.
			elementList_.resize( ( 1 + index / blockSize ) * blockSize, 
				UNKNOWN_NODE );
			elementList_[ index ] = e;
			mainIndex_ = index + 1;
			return 1;
		} else {
			assert( 0 );
			return 0;
		}
	}
}

unsigned int IdManager::findNode( unsigned int index ) const 
{
#ifdef USE_MPI
	Element* e = elementList_[ index ];
	if ( e == 0 || e == UNKNOWN_NODE )
		return BAD_NODE;
	if ( e->cinfo() != initPostMasterCinfo() ) {
		return myNode_;
	} else {
		unsigned int node;
		get< unsigned int >( e, "remoteNode", node );
		return node;
	}
#else
	return 0;
#endif
	
}

/**
 * Returns the most recently created object.
 */
unsigned int IdManager::lastId()
{
	return lastId_;
}

bool IdManager::outOfRange( unsigned int index ) const
{
	return index >= mainIndex_;
}

bool IdManager::isScratch( unsigned int index ) const
{
#ifdef USE_MPI
	return ( myNode_ > 0 && index > 0 && index < scratchIndex_ );
#else
	return 0;
#endif
	
}

/// \todo: Need to put in some grungy code to deal with this.
void IdManager::regularizeScratch()
{
	;
}
