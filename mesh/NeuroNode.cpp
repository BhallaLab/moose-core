/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2012 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
#include "SparseMatrix.h"
#include "Boundary.h"
#include "MeshEntry.h"
#include "VoxelJunction.h"
#include "ChemCompt.h"
#include "MeshCompt.h"
#include "CubeMesh.h"
#include "Vec.h"
#include "CylBase.h"
#include "NeuroNode.h"

/**
 * Helper class for the NeuroMesh. Defines the geometry of the branching
 * neuron.
 */

NeuroNode::NeuroNode( const CylBase& cb, 
		unsigned int parent, const vector< unsigned int >& children,
		unsigned int startFid, Id elecCompt, bool isSphere
   	)
		:
				CylBase( cb ), 
				parent_( parent ),
				children_( children ),
				startFid_( startFid ),
				elecCompt_( elecCompt ),
				isSphere_( isSphere )
{;}

NeuroNode::NeuroNode( Id elecCompt )
		:
				parent_( 0 ),
				startFid_( 0 ),
				elecCompt_( elecCompt ),
				isSphere_( false )
{
	double dia = Field< double >::get( elecCompt, "diameter" );
	setDia( dia );
	double length = Field< double >::get( elecCompt, "length" );
	setLength( length );
	double x = Field< double >::get( elecCompt, "x" );
	double y = Field< double >::get( elecCompt, "y" );
	double z = Field< double >::get( elecCompt, "z" );
	setX( x );
	setY( y );
	setZ( z );
}


NeuroNode::NeuroNode()
		:
				parent_( 0 ),
				startFid_( 0 ),
				elecCompt_( Id() ),
				isSphere_( false )
{;}

unsigned int NeuroNode::parent() const
{
		return parent_;
}

unsigned int NeuroNode::startFid() const
{
		return startFid_;
}

Id NeuroNode::elecCompt() const
{
		return elecCompt_;
}
bool NeuroNode::isDummyNode() const
{
		return ( getNumDivs() == 0 );
}
bool NeuroNode::isSphere() const
{
		return isSphere_;
}
bool NeuroNode::isStartNode() const
{
		return ( startFid_ == 0 );
}

const vector< unsigned int >& NeuroNode::children() const
{
		return children_;
}

void NeuroNode::addChild( unsigned int child )
{
	children_.push_back( child );
}

void NeuroNode::clearChildren()
{
	children_.resize( 0 );
}

void NeuroNode::setParent( unsigned int parent )
{
	parent_ = parent;
}

void NeuroNode::setStartFid( unsigned int fid )
{
	startFid_ = fid;
}

double NeuroNode::calculateLength( const CylBase& parent )
{
	if ( &parent == this ) // Do nothing
			return getLength();
	double dx = parent.getX() - getX();
	double dy = parent.getY() - getY();
	double dz = parent.getZ() - getZ();
	double ret = sqrt( dx * dx + dy * dy + dz * dz );
	setLength( ret );
	return ret;
}

/**
 * Finds all the compartments connected to current node, put them all into
 * the 'children' vector even if they may be 'parent' by the messaging.
 * This is because this function has to be robust enough to sort this out
 */
void NeuroNode::findConnectedCompartments( 
				const map< Id, unsigned int >& nodeMap)
{
	static const Finfo* axialOut = Cinfo::find( "Compartment" )->findFinfo( "axialOut" );
	static const Finfo* raxialOut = Cinfo::find( "Compartment" )->findFinfo( "raxialOut" );
	static const Finfo* distalOut = Cinfo::find( "SymCompartment" )->findFinfo( "distalOut" );
	static const Finfo* proximalOut = Cinfo::find( "SymCompartment" )->findFinfo( "proximalOut" );
	static const Finfo* cylinderOut = Cinfo::find( "SymCompartment" )->findFinfo( "cylinderOut" );
	static const Finfo* sumRaxialOut = Cinfo::find( "SymCompartment" )->findFinfo( "sumRaxialOut" );
	assert( axialOut );
	assert( raxialOut );
	assert( distalOut );
	assert( proximalOut );
	assert( cylinderOut );
	assert( sumRaxialOut );

	const Cinfo* cinfo = elecCompt_.element()->cinfo();
	vector< Id > all;
	if ( cinfo->isA( "SymCompartment" ) ) { // Check derived first.
		vector< Id > ret;
		elecCompt_.element()->getNeighbours( ret, distalOut );
		all.insert( all.end(), ret.begin(), ret.end() );
		elecCompt_.element()->getNeighbours( ret, proximalOut );
		all.insert( all.end(), ret.begin(), ret.end() );
		elecCompt_.element()->getNeighbours( ret, cylinderOut );
		all.insert( all.end(), ret.begin(), ret.end() );
		elecCompt_.element()->getNeighbours( ret, sumRaxialOut );
		all.insert( all.end(), ret.begin(), ret.end() );
	} else {
		assert( cinfo->isA( "Compartment" ) );
		vector< Id > ret;
		elecCompt_.element()->getNeighbours( ret, axialOut );
		all.insert( all.end(), ret.begin(), ret.end() );
		elecCompt_.element()->getNeighbours( ret, raxialOut );
		all.insert( all.end(), ret.begin(), ret.end() );
	}
	sort( all.begin(), all.end() );
	all.erase( unique( all.begin(), all.end() ), all.end() ); //@#$%&* C++
	// Now we have a list of all compartments connected to the current one.
	// Convert to node indices.
	children_.resize( all.size() );
	// Note that the nodeMap only includes compts on list, which may be a
	// subset of compts in entire model. So we only want to explore those.
	for ( unsigned int i = 0; i < all.size(); ++i ) {
		map< Id, unsigned int >::const_iterator k = nodeMap.find( all[i] );
		if ( k != nodeMap.end() )
			children_[i] = k->second;
	}
}


/**
 * Go through nodes vector and eliminate entries that have zero children,
 * that is, are not connected to any others.
 * Need to clean up 'children_' list.
 *
 * static func
 */
unsigned int NeuroNode::removeDisconnectedNodes( 
				vector< NeuroNode >& nodes )
{
	vector< NeuroNode > temp;
	vector< unsigned int > nodeMap( nodes.size() );

	unsigned int j = 0;
	for ( unsigned int i = 0; i < nodes.size(); ++i ) {
		if ( nodes[i].children_.size() > 0 ) {
			temp.push_back( nodes[i] );
			nodeMap[i] = j;
			++j;
		} else {
			nodeMap[i] = ~0;
		}
	}
	for ( unsigned int i = 0; i < temp.size(); ++i ) {
		vector< unsigned int >& c = temp[i].children_;
		for ( vector< unsigned int >::iterator 
						j = c.begin(); j != c.end(); ++j ) {
			assert( nodeMap[ *j ] != ~0U );
			*j = nodeMap[ *j ];
		}
	}
	unsigned int numRemoved = nodes.size() - temp.size();
	nodes = temp;
	return numRemoved;
}

/**
 * Find the start node, typically the soma, of a model. In terms of the
 * solution, this should be the node at the root of the tree. Returns
 * index in nodes vector.
 * Technically the matrix solution could begin from any terminal branch,
 * but it helps to keep the soma identical to the root of the tree.
 *
 * Uses two heuristics to locate the start node: Looks for the node with
 * the largest diameter, and also looks for node(s) with 'soma' in their
 * name. If these disagree then it goes with the 'soma' node. If there are
 * many of the soma nodes, it goes with the fattest.
 *
 * static func
 */
unsigned int NeuroNode::findStartNode( const vector< NeuroNode >& nodes )
{
	double maxDia = 0.0;
	unsigned int somaIndex = ~0;
	for ( unsigned int i = 0; i < nodes.size(); ++i ) {
		const char* name = nodes[i].elecCompt_.element()->getName().c_str();
		if ( strncasecmp( name, "soma", 4 ) == 0 ) {
			if ( maxDia < nodes[i].getDia() ) {
				maxDia = nodes[i].getDia();
				somaIndex = i;
			}
		}
	}
	if ( somaIndex == ~0U ) { // Didn't find any compartment called soma
		for ( unsigned int i =0; i < nodes.size(); ++i ) {
			if ( maxDia < nodes[i].getDia() ) {
				maxDia = nodes[i].getDia();
				somaIndex = i;
			}
		}
	}
	assert( somaIndex != ~0U );
	return somaIndex;
}

/**
 * Traverses the nodes list starting from the 'start' node, and sets up
 * correct parent-child information. This involves removing the 
 * identified 'parent' node from the 'children_' vector and assigning it
 * to the parent_ field.
 * Then it redoes the entire nodes vector (with due care for indexing of
 * children and parents)
 * so that it is in the correct order for a depth-first traversal.
 * This means that you can take any entry in the list, and the immediately
 * following entries will be all the descendants, if any.
 *
 * Static func
 */
void NeuroNode::traverse( vector< NeuroNode >& nodes, unsigned int start )
{
	vector< unsigned int > seen( nodes.size(), ~0 );
	vector< NeuroNode > tree;
	tree.reserve( nodes.size() );
	seen[ start ] = 0;
	tree.push_back( nodes[ start ] );
	tree.back().parent_ = ~0;
	nodes[start].innerTraverse( tree, nodes, seen );

	if ( tree.size() < nodes.size() ) {
		cout << "Warning: NeuroNode::traverse() unable to traverse all nodes:" << tree.size() << " < numNodes = " << nodes.size() << endl;
	}
	nodes = tree;
}

void NeuroNode::innerTraverse( 
				vector< NeuroNode >& tree, 
				const vector< NeuroNode >& nodes,
				vector< unsigned int >& seen 
				) const
{
	unsigned int pa = tree.size() - 1;
	tree.back().children_.clear();

	for ( vector< unsigned int >::const_iterator i = 
					children_.begin(); i != children_.end(); ++i ) {
		assert( *i < nodes.size() );

		// Check that it is an unseen node, ie, not a parent.
		if ( seen[ *i ] == ~0U )  {
			seen[ *i ] = tree.size();
			tree[pa].children_.push_back( tree.size() );
			tree.push_back( nodes[ *i ] );
			tree.back().parent_ = pa;
			nodes[*i].innerTraverse( tree, nodes, seen );
		}
	}
	assert( tree.size() <= nodes.size() );
}

/**
 * This function takes a list of elements that include connected
 * compartments, and constructs a tree of nodes out of them. The 
 * generated nodes vector starts with the soma, and is a depth-first
 * sequence of nodes. This is meant to be insensitive to vagaries
 * in how the user has set up the compartment messaging, provided that 
 * there is at least one recognized message between connected compartments.
 *
 * static function.
 */
void NeuroNode::buildTree( 
				vector< NeuroNode >& nodes, vector< Id > elist )
{
	nodes.clear();
	map< Id, unsigned int > nodeMap;
	for ( vector< Id >::iterator i = elist.begin(); i != elist.end(); ++i )
		if ( i->element()->cinfo()->isA( "Compartment" ) )
			nodes.push_back( NeuroNode( *i ) );
	if ( nodes.size() <= 1 )
		return;
	for ( unsigned int i = 0; i < nodes.size(); ++i )
		nodeMap[ nodes[i].elecCompt() ] = i;
	for ( unsigned int i = 0; i < nodes.size(); ++i )
		nodes[i].findConnectedCompartments( nodeMap );
	removeDisconnectedNodes( nodes );
	unsigned int start = findStartNode( nodes );
	traverse( nodes, start );
}

// Utility function to clean up node indices for parents and children.
void reassignNodeIndices( vector< NeuroNode >& temp, 
				const vector< unsigned int >& nodeToTempMap )
{
	for ( vector< NeuroNode >::iterator 
			i = temp.begin(); i != temp.end(); ++i ) {
		unsigned int pa = i->parent();
		if ( pa != ~0U ) {
			assert( nodeToTempMap[ pa ] != ~0U );
			i->setParent( nodeToTempMap[ pa ] );
		}

	 	vector< unsigned int > kids = i->children();
		i->clearChildren();
		for ( unsigned int j = 0; j < kids.size(); ++j ) {
			unsigned int newKid = nodeToTempMap[ kids[j] ];
			if ( newKid != ~0U ) // Some may be spine shafts, no longer here
				i->addChild( newKid );
		}
	}
}

/**
 * Trims off all spines from tree. Does so by identifying a set of
 * reasonable names: shaft, head, spine, and variants in capitals.
 * Having done this it builds two matching vectors of vector of shafts 
 * and heads, which is a hack that assumes that there are no sub-branches
 * in spines. Then there is an index for parent NeuroNode entry.
 * Static function
 */
void NeuroNode::filterSpines( vector< NeuroNode >& nodes, 
				vector< Id >& shaftId, vector< Id >& headId,
				vector< unsigned int >& parent )
{
	headId.clear();
	shaftId.clear();
	parent.clear();
	vector< NeuroNode > temp;
	temp.reserve( nodes.size() );
	vector< unsigned int > nodeToTempMap( nodes.size(), ~0U );
	vector< unsigned int > shaft;
	vector< unsigned int > reverseShaft( nodes.size(), ~0U );
	vector< unsigned int > head;
	for ( unsigned int i = 0; i < nodes.size(); ++i ) {
		const NeuroNode& n = nodes[i];
		const char* name = n.elecCompt_.element()->getName().c_str();
		if ( strncasecmp( name, "shaft", 5 ) == 0 ||
			strncasecmp( name, "neck", 4 ) == 0 ||
			strncasecmp( name, "spine_neck", 10 ) == 0 ||
			strncasecmp( name, "spine_shaft", 11 ) == 0 ||
			strncasecmp( name, "stalk", 5 ) == 0 ) {
			reverseShaft[i] = shaft.size();
			shaft.push_back( i );
			// Remove from nodes vector by simply not copying.
		} else if ( strncasecmp( name, "spine", 5 ) == 0 ||
			strncasecmp( name, "head", 4 ) == 0 ) {
			head.push_back( i );
			// Remove from nodes vector by simply not copying.
		} else {
			nodeToTempMap[i] = temp.size();
			temp.push_back( n );
		}
	}
	// Now go through finding spine shafts.
	for ( unsigned int i = 0; i < head.size(); ++i ) {
		const NeuroNode& n = nodes[ head[i] ];
		headId.push_back( n.elecCompt() );
		assert( reverseShaft[ n.parent() ] != ~0U );
		const NeuroNode& pa = nodes[ n.parent() ];
		shaftId.push_back( pa.elecCompt() );
		assert( nodeToTempMap[ pa.parent() ] != ~0U );
		parent.push_back( nodeToTempMap[ pa.parent() ] );
	}
	assert( shaftId.size() == headId.size() );

	reassignNodeIndices( temp, nodeToTempMap );
	nodes = temp;
}
