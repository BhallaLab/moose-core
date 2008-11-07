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
#include "shell/Shell.h"
// #include "ArrayWrapperElement.h"
#ifdef USE_MPI
const Cinfo* initPostMasterCinfo();
#endif

const unsigned int UNKNOWN_NODE = UINT_MAX;
// const unsigned int MIN_NODE = 1;
// const unsigned int MAX_NODE = 65536; // Dream on.
const unsigned int IdManager::numScratch = 1000;
const unsigned int IdManager::blockSize = 1000;
const unsigned int BAD_NODE = UINT_MAX;

IdManager::IdManager()
	: loadThresh_( 2000.0 ),
	scratchBegin_( 3 ), // Start at 3 because root is 0 and shell is 1 and postmaster is 2.
	scratchIndex_( 3 ),
	mainIndex_( numScratch )
{
	elementList_.resize( blockSize + numScratch );
}

/*
void IdManager::setNodes( unsigned int myNode, unsigned int numNodes )
{
	myNode_ = myNode;
	numNodes_ = numNodes;
	elementList_[0] = Enode( Element::root(), Shell::myNode() );
	if ( numNodes > 1 ) {
		if ( myNode == 0 )
			nodeLoad.resize( numNodes );
		elementList_.resize( numScratch + blockSize );
		mainIndex_ = numScratch;
	}
}
*/

/**
 * Returns the next available id and allocates space for it.
 * Later can be refined to mop up freed ids. 
 * Don't bother with the scratch space if we are on a single node.
 */
unsigned int IdManager::scratchId()
{
	if ( Shell::numNodes() <= 1 ) {
		lastId_ = mainIndex_;
		mainIndex_++;
		if ( mainIndex_ >= elementList_.size() )
			elementList_.resize( mainIndex_ * 2 );
		return lastId_;
	} else {
		if ( scratchIndex_ < numScratch ) {
			lastId_ = scratchIndex_;
			elementList_[ lastId_ ].setNode( Shell::myNode() );
			++scratchIndex_;
			return lastId_;
		} else {
			cout << "Ran out of scratch Ids on node " << 
				Shell::myNode() << "\n";
			regularizeScratch();
			elementList_[ scratchIndex_ ].setNode( Shell::myNode() );
			return ( lastId_ = scratchIndex_ );
		}
	} 
	return lastId_;
}

// Should only be called on master node, and parent should have been checked
unsigned int IdManager::childId( unsigned int parent )
{
#ifdef USE_MPI
	assert( Shell::myNode() == 0 );
	if ( parent < mainIndex_ ) {
		Enode& pa = elementList_[ parent ];
		if ( mainIndex_ >= elementList_.size() )
			elementList_.resize( elementList_.size() * 2 );
		lastId_ = mainIndex_;
		mainIndex_++;

		if ( parent == 0 ) { // Do load balancing.
			unsigned int targetNode = 
				static_cast< unsigned int >( mainIndex_ / loadThresh_ ) %
				Shell::numNodes();
			elementList_[ lastId_ ] = Enode( 0, targetNode );
		} else if ( pa.node() == Id::GlobalNode ) {
			// Child is also global
			elementList_[ lastId_ ] = Enode( 0, Id::GlobalNode );
		} else {
			// Put object on parent node.
			elementList_[ lastId_ ] = Enode( 0, pa.node() );
		}

		/*
		 * This part does round-robin distribution if the parent element is root,
		 * or if it is on node 0. The above if-else ladder uses round robin only
		 * for the root element's children.
		 */
		//~ if ( parent == 0 || pa.node() == 0 ) { // Do load balancing.
			//~ unsigned int targetNode = 
				//~ static_cast< unsigned int >( mainIndex_ / loadThresh_ ) %
				//~ Shell::numNodes();
			//~ elementList_[ lastId_ ] = Enode( 0, targetNode );
		//~ } else if ( pa.node() == Id::GlobalNode ) {
			//~ // Child is also global
			//~ elementList_[ lastId_ ] = Enode( 0, Id::GlobalNode );
		//~ } else if ( pa.node() != Shell::myNode() ) {
			//~ // Put object on parent node.
			//~ elementList_[ lastId_ ] = Enode( 0, pa.node() );
		//~ } else {
			//~ assert( 0 );
		//~ }
		
		return lastId_;
	}
	assert( 0 );
#else
	lastId_ = mainIndex_;
	mainIndex_++;
	if ( mainIndex_ >= elementList_.size() )
		elementList_.resize( mainIndex_ * 2 );
	return lastId_;
#endif
}

/**
 * This variant of childId forces creation of object on specified node,
 * provided that we are in parallel mode. Otherwise it ignores the node
 * specification.  
 * Should only be called on master node, and parent should have been checked
 * In single-node mode it is equivalent to the scratchId and childId ops.
 * \todo Need to add facility to create object globally.
 *
 */
unsigned int IdManager::makeIdOnNode( unsigned int childNode )
{
	lastId_ = mainIndex_;
	mainIndex_++;
#ifdef USE_MPI
	assert( Shell::myNode() == 0 );
	assert( childNode < Shell::numNodes() );
	elementList_[ lastId_ ] = Enode( 0, childNode );
#endif
	if ( mainIndex_ >= elementList_.size() )
		elementList_.resize( mainIndex_ * 2 );
	return lastId_;
}

Element* IdManager::getElement( const Id& id ) const
{
	if ( id.id() < mainIndex_ ) {
		const Enode& ret = elementList_[ id.id() ];
#ifdef USE_MPI
		if ( ret.node() == UNKNOWN_NODE ) {
			// don't know how to handle this yet. It should trigger
			// a request to the master node to update the elist.
			// We then get into managing how many entries are unknown...
			assert( 0 );
			return 0;
		}
		/*
		} else if ( ret.node() == Shell::myNode() || 
			ret.node() == Id::GlobalNode ) {
			return ret.e();
		} else {
			return 0;
		}
		*/
#endif
		return ret.e();
	}
	return 0;
}

/**
 * All this does is assign the element. The node must be assigned at
 * the time the id is created.
 */
bool IdManager::setElement( unsigned int index, Element* e )
{
	if ( index >= elementList_.size() )
		elementList_.resize( ( 1 + index / blockSize ) * blockSize );

	if ( index < numScratch ) {
		elementList_[ index ].setElement( e );
		return 1;
	}

	if ( index < mainIndex_ ) {
		Enode& old = elementList_[ index ];
		if ( old.node() == UNKNOWN_NODE || old.e() == 0 ) {
			elementList_[ index ].setElement( e );
			// = Enode( e, Shell::myNode() );
			return 1;
		} else if ( e == 0 ) {
			// Here we are presumably clearing out an element. Permit it.
			elementList_[ index ] = Enode( 0, Shell::myNode() );
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
		// Here we have been told by the master node to make a child 
		// at a specific index before the elementList has been
		// expanded to that index. Just expand it to fit.
		elementList_[ index ].setElement( e );
		// = Enode( e, Shell::myNode() );
		mainIndex_ = index + 1;
		return 1;
	}
}

unsigned int IdManager::scratchIndex() const
{
	return scratchIndex_;
}

bool IdManager::redefineScratchIds( unsigned int last,
	unsigned int base, unsigned int node )
{
	// Problem here, if we spill over the scratchIndex size.
	// May well happen for big cells. Need to resolve.
	assert( last < scratchIndex_ );
	unsigned int num = scratchIndex_ - last;
	if ( num + base >= elementList_.size() )
		elementList_.resize( ( 1 + (num + base) / blockSize ) * blockSize );
	for ( unsigned int i = last; i < scratchIndex_; ++i ) {
		Enode& en = elementList_[ i ];
		elementList_[ base ] = Enode( en.e(), node );
		en.e()->setId( Id( base, 0 ) );
		++base;
		en = Enode();
	}
	return 1;
}


#ifdef USE_MPI
unsigned int IdManager::findNode( unsigned int index ) const 
{
	if ( index == Id::badId().id() )
		return BAD_NODE;
	
	assert( index < elementList_.size() );
	const Enode& e = elementList_[ index ];
	if ( e.node() == UNKNOWN_NODE )
		return BAD_NODE;
	
	return e.node();
}

// Do not permit op if parent is not global
void IdManager::setGlobal( unsigned int index )
{
	assert( index < elementList_.size() );
	Enode& e = elementList_[ index ];
	e.setGlobal();
}

bool IdManager::isGlobal( unsigned int index ) const 
{
	assert( index < elementList_.size() );
	const Enode& e = elementList_[ index ];
	return ( e.node() == Id::GlobalNode );
}

void IdManager::setNode( unsigned int index, unsigned int node )
{
	assert( node < Shell::numNodes() || node == Id::GlobalNode );
	
	if ( index >= elementList_.size() )
		elementList_.resize( index * 2 );
	
	Enode& e = elementList_[ index ];
	// cout << "Setting node for " << index << " to " << node << endl;
	e.setNode( node );
}

#else
unsigned int IdManager::findNode( unsigned int index ) const 
{
	return 0;
}

bool IdManager::isGlobal( unsigned int index ) const 
{
	return 0;
}

void IdManager::setGlobal( unsigned int index )
{
	;
}

void IdManager::setNode( unsigned int index, unsigned int node )
{
	;
}
#endif


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
	return ( Shell::myNode() > 0 && index > 0 && index < scratchIndex_ );
#else
	return 0;
#endif
	
}

/// \todo: Need to put in some grungy code to deal with this.
void IdManager::regularizeScratch()
{
scratchBegin_ = 14;
	unsigned int numPromote = scratchIndex_ - scratchBegin_;
	if ( scratchIndex_ == numScratch )
		numPromote--;
	scratchIndex_ = scratchBegin_;

	unsigned int baseId;
	if ( Shell::myNode() == 0 )
		baseId = mainIndex_;
	else
		baseId = Shell::regularizeScratch( numPromote );
	assert( baseId >= numScratch );

	unsigned int id = baseId;
	vector< Enode >::iterator scratch =
		elementList_.begin() + scratchBegin_;
	for ( unsigned int i = 0; i < numPromote; i++ ) {
		scratch->e()->setId( Id( id ) );
		
		setElement( id, scratch->e() );
		elementList_[ id ].setNode( Shell::myNode() );
		
		if( isGlobal( id ) ) {
			// inform other nodes
		}
		
		*scratch = Enode( 0, 0 );
		
		id++, scratch++;
	}
}

unsigned int IdManager::allotMainIdBlock( unsigned int size, unsigned int node )
{
	assert( Shell::myNode() == 0 );

	lastId_ = mainIndex_;
	mainIndex_ += size;
	if ( mainIndex_ >= elementList_.size() )
		elementList_.resize( mainIndex_ * 2 );

	//~ If the alloted space needs to be filled with something.
	//~ Enode e( Id::postId( node )(), node );
	//~ fill(
		//~ elementList_.begin() + lastId_,
		//~ elementList_.begin() + lastId_ + size,
		//~ e
	//~ );

	return lastId_;
}
