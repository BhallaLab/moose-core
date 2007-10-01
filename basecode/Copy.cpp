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
	const Element* parent = connDestBegin( 0 )->targetElement();
	if ( parent == ancestor )
		return 1;
	else
		return parent->isDescendant( ancestor);
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
	SimpleElement* ret = new SimpleElement( this );
	
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

Element* SimpleElement::innerCopy(int n) const
{	
	assert( finfo_.size() > 0 );
	assert( dynamic_cast< ThisFinfo* >( finfo_[0] ) != 0 );
	void *data = finfo_[0]->ftype()->copyIntoArray( data_, 1, n );
	ArrayElement* ret = new ArrayElement( name_, src_, dest_, conn_, finfo_, data, n, 0 );
	//cout <<  "IDS "<< ret->id() << " " << id() << endl;
	//cout << (ret->id())()->id() << endl;
	//ret->CopyFinfosSimpleToArray(this);
	return ret;
}

//obsolete::delete it 
void ArrayElement::CopyFinfosSimpleToArray(const SimpleElement *se){
	vector< const Finfo* > flist;
	se->listFinfos(flist);
	for (size_t i = 0; i < flist.size(); i++ ){
		//cout << i << " " << flist.size() << endl;
		this->finfo_.push_back(flist[i]->copy());
	}
}


/**
 * This function fills up the map with current element and
 * all its descendants. Returns the root element of the copied tree.
 * The first entry in the map is the original
 * The second entry in the map is the copy.
 * The function does NOT fix up the messages.
 *
 * The function has to be careful if the element is a global. When
 * the global occurs at the root of the tree, it is assumed that we
 * really do want to make a copy. Otherwise it is bunged into the
 * tree itself and not copied, but later clever things are done with
 * the messaging.
 */
Element* SimpleElement::innerDeepCopy(
	map< const Element*, Element* >& tree ) const
{
	if ( isGlobal() && tree.size() >= 0 ) {
		Element* cme = const_cast< SimpleElement* >( this );
		// RDWORRY: What about the ArrayElement*?
		tree[ this ] = cme;
		return cme;
	}

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

Element* SimpleElement::innerDeepCopy(
	map< const Element*, Element* >& tree, int n ) const
{
	if ( isGlobal() && tree.size() >= 0 ) {
		Element* cme = const_cast< SimpleElement* >( this );
		// RDWORRY: What about the ArrayElement*?
		tree[ this ] = cme;
		return cme;
	}

	Element* duplicate = innerCopy(n);
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
			i->targetElement()->innerDeepCopy( tree, n );
	}
	return duplicate;
}

/**
 * This function does a deep copy of the current element
 * including all messages. Returns the base of the copied tree.
 * It attaches the copied element tree to the parent.
 * It renames the copied tree base if the newName is not empty.
 * It first checks that the parent does not already have a child
 * of the target name.
 * It is non-recursive but calls lots recursive functions.
 *
 * A special case happens if one of the source elements in the tree
 * has the isGlobal flag. This flag means that the source, typically
 * an object on the library, is not to be duplicated but instead
 * its messages into the tree should be duplicate. This is used,
 * for example, for HHGates on HHChannels. All channel duplicates
 * should use the same HHGates.
 * The global elements are put into the tree as their own copy.
 * A further refinement is that global elements can be copied but only
 * if they are the root of the copy tree. In this case we assume that
 * the user really does want to copy the global element as they have
 * specifically requested the copy.
 *
 * \todo: This will need a lot of work to handle cross-node copies,
 * or even worse, copies of element trees that span nodes. For now
 * it is single-node stuff.
 */

Element* SimpleElement::copy( Element* parent, const string& newName )
		const
{
	static const Element* library = Id( "/library" )();
	static const Element* proto = Id( "/proto" )();

	if ( parent->isDescendant( this ) ) {
		cout << "Warning: SimpleElement::copy: Attempt to copy within descendant tree" << parent->name() << endl;
		return 0;
	}
	string nm = newName;

	if ( newName == "" )
		nm = name();
	Id oldChild;
	bool ret = lookupGet< Id, string >(
					parent, "lookupChild", oldChild, nm );
	assert( ret );
	if ( !oldChild.bad() ) {
		cout << "Warning: SimpleElement::copy: pre-existing child with target name: " << parent->name() << "/" << nm << endl;
		return 0;
	}

	// First is original, second is copy
	map< const Element*, Element* > tree;
	map< const Element*, Element* >::iterator i;
	vector< pair< Element*, unsigned int > > delConns;

	Element* child = innerDeepCopy( tree );
	child->setName( nm );

	// First pass: Replace copy pointers so that the dup is set up right
	for ( i = tree.begin(); i != tree.end(); i++ ) {
		if ( i->first != i->second ) {
			i->second->replaceCopyPointers( tree, delConns );
		}
	}

	// Second pass: Delete any outgoing messages or messages to globals
	vector< pair< Element*, unsigned int > >::iterator j;
	for ( j = delConns.begin(); j != delConns.end(); j++ )
		j->first->deleteHalfConn( j->second );
	
	// Third pass: Copy over messages to any global elements.
	for ( i = tree.begin(); i != tree.end(); i++ ) {
		if ( i->first == i->second ) { // a global
			i->second->copyMsg( tree );
		}
	}
	
	// Fourth pass: stick the copied tree onto the parent Element.
	ret = parent->findFinfo( "childSrc" )->add(
					parent, child, child->findFinfo( "child" ) );
	assert( ret );

	// Fifth pass: Schedule all the objects
	if ( !( 
		parent->isDescendant( library ) || parent->isDescendant( proto )
		) ) {
		for ( i = tree.begin(); i != tree.end(); i++ ) {
			if ( i->first != i->second ) // a global
				i->second->cinfo()->schedule( i->second );
		}
	}

	return child;
}

Element* SimpleElement::copyIntoArray( Element* parent, const string& newName, int n )
		const
{
	static const Element* library = Id( "/library" )();
	static const Element* proto = Id( "/proto" )();

	if ( parent->isDescendant( this ) ) {
		cout << "Warning: SimpleElement::copy: Attempt to copy within descendant tree" << parent->name() << endl;
		return 0;
	}
	string nm = newName;

	if ( newName == "" )
		nm = name();
	Id oldChild;
	//oldChild = Neutral::getChildByName( parent, nm)
	bool ret = lookupGet< Id, string >(
					parent, "lookupChild", oldChild, nm );
	assert( ret );
	if ( !oldChild.bad() ) {
		cout << "Warning: SimpleElement::copy: pre-existing child with target name: " << parent->name() << "/" << nm << endl;
		return 0;
	}

	// First is original, second is copy
	map< const Element*, Element* > tree;
	map< const Element*, Element* >::iterator i;
	vector< pair< Element*, unsigned int > > delConns;

	Element* child = innerDeepCopy( tree, n );
	child->setName( nm );

	// First pass: Replace copy pointers so that the dup is set up right
	for ( i = tree.begin(); i != tree.end(); i++ ) {
		if ( i->first != i->second ) {
			i->second->replaceCopyPointers( tree, delConns );
		}
	}

	// Second pass: Delete any outgoing messages or messages to globals
	vector< pair< Element*, unsigned int > >::iterator j;
	for ( j = delConns.begin(); j != delConns.end(); j++ )
		j->first->deleteHalfConn( j->second );
	
	// Third pass: Copy over messages to any global elements.
	for ( i = tree.begin(); i != tree.end(); i++ ) {
		if ( i->first == i->second ) { // a global
			i->second->copyMsg( tree );
		}
	}
	
	// Fourth pass: stick the copied tree onto the parent Element.
	ret = parent->findFinfo( "childSrc" )->add(parent, child, child->findFinfo( "child" ) );
	assert( ret );

	// Fifth pass: Schedule all the objects
	if ( !( 
		parent->isDescendant( library ) || parent->isDescendant( proto )
		) ) {
		for ( i = tree.begin(); i != tree.end(); i++ ) {
			if ( i->first != i->second ) // a global
				i->second->cinfo()->schedule( i->second );
		}
	}
	return child;
}

/**
 * Takes all the messages between this element and the
 * key (original) portion of the tree, and duplicate them to go
 * between the current element and the data (copied) portion of
 * the tree. Exclude child messages.
 * Note that this is about the
 * opposite of the criterion for halo messages, which are all messages
 * except those going into the tree.
 */
void SimpleElement::copyMsg( map< const Element*, Element* >& tree ) 
{
	unsigned int i;
	vector< Conn >::iterator j;
	map< const Element*, Element* >::iterator k;
	// Skip the child connection, so start at 1.
	for ( i = 1; i < connSize(); i++ ) {
		j = lookupVariableConn( i );
		k = tree.find( j->targetElement() );
		if ( k != tree.end() )
			this->innerCopyMsg( *j, k->first, k->second );
	}
}

/**
 * Coming into this function, the Conn c goes between this and orig.
 * When it is done, a new Conn is set up between this and the
 * duplicated Element. Returns true if addmsg worked.
 * Still need to check if it handles all cases of src/dest finfos.
 */
bool SimpleElement::innerCopyMsg(
	Conn& c, const Element* orig, Element* dup )
{
	assert( orig != dup );
	assert( orig->className() == dup->className() );
	// Start out by trying to find Finfo on local element
	// We don't know yet if this is msgsrc or msgdest.
	const Finfo* temp = findFinfo( connIndex( &c ) );
	assert( temp );
	if ( temp->name() == "child" || temp->name() == "childSrc" )
		return 0;
	const Finfo* srcFinfo;
	const Finfo* destFinfo;
	if ( dynamic_cast< const SrcFinfo* >( temp ) != 0 ) {
		srcFinfo = temp;
		destFinfo = orig->findFinfo( c.targetIndex() );
		return srcFinfo->add( this, dup, destFinfo );
	} else {
		srcFinfo = orig->findFinfo( c.targetIndex() );
		destFinfo = temp;
		return srcFinfo->add( dup, this, destFinfo );
	}
	return 0;
}

/**
 * This function replaces Element* pointers in the conn_ vector
 * with corresponding ones from the copied tree.
 *
 * While doing so it also fills out a vector to keep track of conns
 * that need to be deleted. This vector has to be filled out now,
 * because the information on targetElement can only be found from the
 * tree before the targets are replaced here. However, the deletion of
 * the conns cannot be done till the entire tree has had its targets
 * replaces. So the vector is filled here and executed later.
 *
 * In the case of the HaloCopy (yet to be implemented ) we would
 * instead create new exterior conns to duplicate the linkages
 * of the old ones.
 */

void SimpleElement::replaceCopyPointers(
	map< const Element*, Element* >& tree,
	vector< pair< Element*, unsigned int > >& delConns )
{
	if ( conn_.size() == 0 ) return;
	map< const Element*, Element* >::iterator j;
	
	unsigned int i = conn_.size();
	while ( i > 0 ) {
		--i;
		j = tree.find( conn_[ i ].targetElement() );
		if ( j != tree.end() ) { // Inside the tree. Just replace ptrs.
			if ( j->first != j->second ) // Don't mess with globals.
				conn_[ i ].replaceElement( j->second );
			else // Globals must have their conns deleted
				delConns.push_back( 
					pair< Element*, unsigned int >( this, i ) );
		} else { // Objects outside the tree must have their conns deleted
			delConns.push_back( 
				pair< Element*, unsigned int >( this, i ) );
		}
	}
}

void ArrayElement::replaceCopyPointers(
	map< const Element*, Element* >& tree,
	vector< pair< Element*, unsigned int > >& delConns )
{
	if ( conn_.size() == 0 ) return;
	map< const Element*, Element* >::iterator j;
	
	unsigned int i = conn_.size();
	while ( i > 0 ) {
		--i;
		j = tree.find( conn_[ i ].targetElement() );
		if ( j != tree.end() ) { // Inside the tree. Just replace ptrs.
			if ( j->first != j->second ) // Don't mess with globals.
				conn_[ i ].replaceElement( j->second );
			else // Globals must have their conns deleted
				delConns.push_back( 
					pair< Element*, unsigned int >( this, i ) );
		} else { // Objects outside the tree must have their conns deleted
			delConns.push_back( 
				pair< Element*, unsigned int >( this, i ) );
		}
	}
}

#ifdef DO_UNIT_TESTS

void copyTest()
{
	cout << "\nTesting copy";

	Element* n = Neutral::create( "Neutral", "n", Element::root() );
	Element* c0 = Neutral::create( "Compartment", "c0", n );
	ASSERT( c0 != 0, "creating compartment" );

	Element* ch = Neutral::create( "HHChannel", "ch", c0 );
	ASSERT( ch != 0, "creating channel" );
	
	ASSERT(
		c0->findFinfo( "channel" )->
		add( c0, ch, ch->findFinfo( "channel" ) ), "set up copy" );
	set< double >( ch, "Xpower", 3.0 );
	set< double >( ch, "Ypower", 1.0 );


	ProcInfoBase p;
	Conn c( c0, 0 );
	p.dt_ = 0.002;
	set< double >( c0, "inject", 1.0 );
	set< double >( c0, "Rm", 2.0 );
	set< double >( c0, "Ra", 3.0 );
	set< double >( c0, "Cm", 4.0 );
	set< double >( c0, "Em", 5.0 );
	set< double >( c0, "Vm", 6.0 );

	Element* c1 = c0->copy( n, "c1" );
	ASSERT( c1 != c0, "copying" );
	ASSERT( c1 != 0, "copying" );

	ASSERT( c1->name() == "c1", "copying" );

	// Check copied values
	double dret;
	get< double >( c1, "inject", dret );
	ASSERT( dret == 1.0, "copying" );
	get< double >( c1, "Rm", dret );
	ASSERT( dret == 2.0, "copying" );
	get< double >( c1, "Ra", dret );
	ASSERT( dret == 3.0, "copying" );
	get< double >( c1, "Cm", dret );
	ASSERT( dret == 4.0, "copying" );
	get< double >( c1, "Em", dret );
	ASSERT( dret == 5.0, "copying" );
	get< double >( c1, "Vm", dret );
	ASSERT( dret == 6.0, "copying" );

	// Check that copy is a unique object
	bool ret;
	ret = set< double >( c1, "inject", 0.1 );
	ASSERT( ret, "copy uniqueness" );
	get< double >( c0, "inject", dret );
	ASSERT( dret == 1.0, "copying" );
	get< double >( c1, "inject", dret );
	ASSERT( dret == 0.1, "copying" );

	// Check copied tree
	Element* e = Neutral::getChildByName( c1, "ch" )();
	ASSERT( e != 0, "copied child" );
	get< double >( e, "Xpower", dret );
	ASSERT( dret == 3.0, "copying" );


	//////////////////////////////////////////////////////////////////
	// Test out copies when there is a global element in the tree.
	// The copy should have messages to the original global element.
	//////////////////////////////////////////////////////////////////
	Element* temp = Neutral::getChildByName( e, "xGate" )( );
	ASSERT( temp == 0, "copied global child: should not exist" );
	temp = Neutral::getChildByName( e, "yGate" )( );
	ASSERT( temp == 0, "copied global child: should not exist" );

	// See if the messages go to the original Gates.
	vector< Conn > clist;
	const Finfo* xGateFinfo = ch->findFinfo( "xGate" );
	unsigned int numConns = xGateFinfo->incomingConns( ch, clist );
	ASSERT( numConns == 1, "Original connections on x gate" );

	temp = Neutral::getChildByName( ch, "xGate" )( );
	ASSERT( temp != 0, "original xgate" );
	ASSERT( temp == clist[0].targetElement(), "Testing orig gate" );

	numConns = xGateFinfo->incomingConns( e, clist );
	// unsigned int nc2 = xGateFinfo->outgoingConns( e, clist );
	ASSERT( numConns == 1, "Shifted connections on x gate" );
	ASSERT( temp == clist[0].targetElement(), "Testing that gate msg is to same place as orig gate" );

	temp = Neutral::getChildByName( ch, "yGate" )( );
	ASSERT( temp != 0, "original ygate" );
	const Finfo* yGateFinfo = ch->findFinfo( "yGate" );
	numConns = yGateFinfo->incomingConns( ch, clist );
	ASSERT( numConns == 1, "Original connections on y gate" );
	ASSERT( temp == clist[0].targetElement(), "Testing orig gate" );

	numConns = yGateFinfo->incomingConns( e, clist );
	ASSERT( numConns == 1, "Shifted connections on y gate" );
	ASSERT( temp == clist[0].targetElement(), "Testing that gate msg is to same place as orig gate" );


	
	//////////////////////////////////////////////////////////////////
	// Check copy preserving old name. Copy c0 onto c1.
	//////////////////////////////////////////////////////////////////
	Element* c10 = c0->copy( c1, "" );
	ASSERT( c10 != c0, "copying" );
	ASSERT( c10 != 0, "copying" );
	ASSERT( c10->name() == "c0", "copying" );

	// Check that the copy has a unique id (this was an actual bug!)
	ASSERT( c10->id() != c0->id(), "unique copy id" );

	

	set( n, "destroy" );
}

#endif // DO_UNIT_TESTS
