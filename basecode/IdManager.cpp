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
#include "shell/Shell.h"
#ifdef USE_MPI
const Cinfo* initPostMasterCinfo();
#endif

const unsigned int IdManager::initBlockEnd = 15;
const unsigned int IdManager::blockSize = 1000;
// const unsigned int MIN_NODE = 1;
// const unsigned int MAX_NODE = 65536; // Dream on.

IdManager::IdManager()
	: loadThresh_( 2000.0 ),
	initIndex_( 3 ) // Start at 3 because root is 0 and shell is 1 and postmaster is 2.
{
	const unsigned int numNodes = Shell::numNodes();
	const unsigned int myNode = Shell::myNode();

	const unsigned int firstBlockSize = numNodes;
	const unsigned int lastNode = numNodes - 1;

	localIndex_ = initBlockEnd + firstBlockSize * ( lastNode - myNode );
	blockEnd_ = localIndex_ + firstBlockSize;

	elementList_.resize( blockEnd_ + blockSize );
	fill(
		elementList_.begin() + localIndex_,
		elementList_.begin() + blockEnd_,
		Enode( 0, Shell::myNode() )
	);
}

/**
 * \todo Figure out how to destroy elements at exit.
 */
IdManager::~IdManager()
{
	//~ Crude attempt to destroy elements.
	//~ vector< Enode >::iterator ie;
	//~ for ( ie = elementList_.begin(); ie != elementList_.end(); ie++ ) {
		//~ Element* el = ie->e();
		//~ if ( el != 0 )
			//~ set( el, "destroy" );
	//~ }
}

void IdManager::dumpState( ostream& stream ) {
	unsigned int i;
	unsigned int zeroBegin;
	bool zeroRun = false;

	const unsigned int numNodes = Shell::numNodes();
	const unsigned int myNode = Shell::myNode();

	const unsigned int firstBlockSize = numNodes;
	const unsigned int lastNode = numNodes - 1;
	stream << initBlockEnd + firstBlockSize * ( lastNode - myNode ) << " - " << blockEnd_ - 1 << "\n";

	for ( i = 0; i < blockEnd_; i++ ) {
		Element* e = elementList_[ i ].e();
		if ( e != 0 ) {
			if ( zeroRun ) {
				stream << zeroBegin << " - " << i - 1 << ": 0x0\n";
				zeroRun = false;
			}
			stream << i << ": " << e->name() << "\n";
		} else if ( ! zeroRun ) {
			zeroRun = true;
			zeroBegin = i;
		}
	}

	stream << flush;
}

unsigned int IdManager::newId() {
	assert( localIndex_ <= blockEnd_ );
	
	if ( localIndex_ == blockEnd_ ) { // Local pool exhausted.
		if ( Shell::myNode() == 0 )
			localIndex_ = newIdBlock( blockSize, Shell::myNode() );
		else
			localIndex_ = Shell::newIdBlock( blockSize );
		
		blockEnd_ = localIndex_ + blockSize;
		
		if ( blockEnd_ >= elementList_.size() )
			elementList_.resize( 2 * blockEnd_ );
		
		fill(
			elementList_.begin() + localIndex_,
			elementList_.begin() + blockEnd_,
			Enode( 0, Shell::myNode() )
		);
	}
	
	lastId_ = localIndex_;
	localIndex_++;
	return lastId_;
}

unsigned int IdManager::newIdBlock( unsigned int size, unsigned int node )
{
	assert( Shell::myNode() == 0 );
	
	/*
	 * Here blockBegin and blockEnd are the boundaries of the block assigned to
	 * the requesting node. On the other hand, blockEnd_ (with the underscore)
	 * is the end of the master node's block.
	 */
	unsigned int blockBegin = localIndex_;
	unsigned int blockEnd = blockBegin + size;
	
	localIndex_ += size;
	blockEnd_ += size;
	
	if ( blockEnd_ >= elementList_.size() )
		elementList_.resize( 2 * blockEnd_ );
	
	fill(
		elementList_.begin() + blockBegin,
		elementList_.begin() + blockEnd,
		Enode( 0, node )
	);
	
	return blockBegin;
}

IdGenerator IdManager::generator( unsigned int node )
{
	unsigned int id = Id::badId().id();
	if ( node == Id::GlobalNode )
		id = localIndex_;
	return IdGenerator( id, node );
}

unsigned int IdManager::initId() {
	assert( initIndex_ < initBlockEnd );
	lastId_ = initIndex_;
	initIndex_++;
	return lastId_;
}

// Should only be called on master node, and parent should have been checked
unsigned int IdManager::childNode( unsigned int parent )
{
#ifdef USE_MPI
	assert( Shell::myNode() == 0 );
	assert( parent < localIndex_ );
	
	unsigned int childNode;
	Enode& pa = elementList_[ parent ];
	
	if ( parent == 0 ) { // Do load balancing.
		childNode =	static_cast< unsigned int >
			( localIndex_ / loadThresh_ ) % Shell::numNodes();
	} else if ( pa.node() == Id::GlobalNode ) {
		// Child is also global
		childNode = Id::GlobalNode;
	} else {
		// Put object on parent node.
		childNode = pa.node();
	}
	
	return childNode;
#else
	return 0;
#endif
}

// Should only be called on master node, and parent should have been checked
unsigned int IdManager::childId( unsigned int parent )
{
	return makeIdOnNode( childNode( parent ) );
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
	unsigned int childId = newId();
#ifdef USE_MPI
	assert( Shell::myNode() == 0 );
	assert( childNode == Id::GlobalNode || childNode < Shell::numNodes() );
	
	elementList_[ childId ] = Enode( 0, childNode );
#endif
	return childId;
}

Element* IdManager::getElement( const Id& id ) const
{
	assert( id.id() < elementList_.size() );
	
	if ( ! id.outOfRange() ) {
		const Enode& ret = elementList_[ id.id() ];
#ifdef USE_MPI
		if ( ret.node() == Id::UnknownNode ) {
			// don't know how to handle this yet. It should trigger
			// a request to the master node to update the elist.
			// We then get into managing how many entries are unknown...
			assert( 0 );
			return 0;
		}
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

	Enode& old = elementList_[ index ];
	if ( old.node() == Id::UnknownNode || old.e() == 0 ) {
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
}

#ifdef USE_MPI
unsigned int IdManager::findNode( unsigned int index ) const 
{
	if ( index == Id::badId().id() )
		return Id::BadNode;
	
	assert( index < elementList_.size() );
	const Enode& e = elementList_[ index ];
	if ( e.node() == Id::UnknownNode )
		return Id::BadNode;
	
	return e.node();
}

// Do not permit op if parent is not global
void IdManager::setGlobal( unsigned int index )
{
	if ( index >= elementList_.size() )
		elementList_.resize( ( 1 + index / blockSize ) * blockSize );

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
		elementList_.resize( ( 1 + index / blockSize ) * blockSize );
	
	Enode& e = elementList_[ index ];
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
unsigned int IdManager::lastId() const
{
	return lastId_;
}

bool IdManager::outOfRange( unsigned int index ) const
{
	return ( index >= elementList_.size() );
}
