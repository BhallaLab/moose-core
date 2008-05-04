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

	Conn* c = targets( "child" );
	assert( c->good() ); // It better have a parent!
	const Element* parent = c->target().e;
	delete c;

	if ( parent == ancestor )
		return 1;
	else
		return parent->isDescendant( ancestor);
}

/**
 * This function copies the element, its data and its dynamic Finfos.
 * It also copies the Msg vector over. This includes the original
 * ConnTainer vector and its pointers, which now have to be fixed.
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
		Finfo* temp = finfo_[i]->copy();
		// assert( ret->finfo_[i] == 0 );
		// ret->finfo_[i] = finfo_[i]->copy();
		assert( temp != 0 );
		assert( temp != finfo_[i] );
		// Sometimes dynamicFinfos will have messages within the tree,
		// or there will be a 'next' message. We'll fill in the 'next'
		// messages later, as needed, but the dynamicFinfo msg slots
		// need to be set up.
		ret->addFinfo( temp );
	}
	return ret;
}

Element* SimpleElement::innerCopy(int n) const
{	
	return 0;
	/*
	assert( finfo_.size() > 0 );
	assert( dynamic_cast< ThisFinfo* >( finfo_[0] ) != 0 );
	// void *data = finfo_[0]->ftype()->copyIntoArray( data_, 1, n );
	ArrayElement* ret = new ArrayElement( name_, src_, dest_, conn_, finfo_, data, n, 0 );
	//cout <<  "IDS "<< ret->id() << " " << id() << endl;
	//cout << (ret->id())()->id() << endl;
	//ret->CopyFinfosSimpleToArray(this);
	return ret;
	*/
}


/**
 * This function fills up the map with current element and
 * all its descendants. Returns the root element of the copied tree.
 * The first entry in the map is the original
 * The second entry in the map is the copy.
 * The function does NOT fix up the messages. 
 *
 * All messages, even parent-child messages, are blank in the copy.
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
	static unsigned int childSrcMsg = 
		initNeutralCinfo()->getSlot( "childSrc" ).msg();

	assert ( childSrcMsg == 0 );

	if ( isGlobal() && tree.size() >= 0 ) {
		Element* cme = const_cast< SimpleElement* >( this );
		// RDWORRY: What about the ArrayElement*?
		tree[ this ] = cme;
		return cme;
	}

	Element* duplicate = innerCopy();
	tree[ this ] = duplicate;

	const Msg* childMsg = msg( childSrcMsg );
	assert( childMsg != 0 );
	if ( childMsg->size() > 0 )
		assert( childMsg->isDest() == 0 );

	/**
	 * Note that we iterate through ConnTainers here. Each unique child,
	 * whether simple or array, is stored in an individual ConnTainer.
	 */
	vector< ConnTainer* >::const_iterator i;
	for ( i = childMsg->begin(); i != childMsg->end(); i++ ) {
		Element* tgt = ( *i )->e2();
		if ( tree.find( tgt ) != tree.end() )
			cout << "Warning: SimpleElement::innerDeepCopy: Loop in element tree at " << tgt->name() << endl;
		else 
			tgt->innerDeepCopy( tree );
	}
	return duplicate;
}

Element* SimpleElement::innerDeepCopy(
	map< const Element*, Element* >& tree, int n ) const
{
	return 0;
/*
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
	vector< Conn >::const_iterator end = connSrcVeryEnd( 0 );
	for ( i = begin; i != end; i++ ) {
		// Watch out for loops.
		if ( tree.find( i->targetElement() ) != tree.end() )
			cout << "Warning: SimpleElement::innerDeepCopy: Loop in element tree at " << i->targetElement()->name() << endl;
		else 
			i->targetElement()->innerDeepCopy( tree, n );
	}
	return duplicate;
	*/
}

/**
 * Copies messages from current element to duplicate provided dest is
 * also on tree.
 */
void SimpleElement::copyMessages( Element* dup, 
	map< const Element*, Element* >& origDup ) const
{
	map< const Element*, Element* >::iterator k;
	vector< Msg >::const_iterator m;
	// This assertion fails because numMsg() may be incremented
	// to accommodate 'next' msgs and dynamicFinfos.
	// assert( dup->numMsg() == numMsg() );
	for ( m = msg_.begin(); m != msg_.end(); m++ ) {
		if ( m->size() == 0 )
			continue;
		if ( m->isDest() )
			continue;
		vector< ConnTainer* >::const_iterator c;
		for ( c = m->begin(); c != m->end(); c++ ) {
			k = origDup.find( ( *c )->e2() );
			if ( k != origDup.end() ) {
				m->copy( *c, dup, k->second );
			}
		}
	}
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
 * its messages into the tree should be duplicated. This is used,
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
	// Phase 0: Set up and check stuff for the copy.
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

	// Phase 1. Copy Elements, but not building up parent-child info.
	// First is original, second is copy
	// However, if it was a Global, both original and second are the same.
	map< const Element*, Element* > origDup;
	map< const Element*, Element* >::iterator i;

	vector< pair< Element*, unsigned int > > delConns;

	Element* child = innerDeepCopy( origDup );
	child->setName( nm );

	// Phase 2. Copy over messages that are within the tree.
	// Here we need only copy from message sources.
	for ( i = origDup.begin(); i != origDup.end(); i++ ) {
		if ( i->first != i->second ) {
			i->first->copyMessages( i->second, origDup );
		}
	}

	
	// Phase 3 : Copy over messages to any global elements that were
	// not on the original tree.
	// Still to fill in.
	
	// Phase 4: stick the copied tree onto the parent Element.
	ret = Eref( parent ).add( "childSrc", child, "child", 
		ConnTainer::Default );
	/*
	ret = parent->findFinfo( "childSrc" )->add(
					parent, child, child->findFinfo( "child" ) );
					*/
	assert( ret );

	// Phase 5: Schedule all the objects
	if ( !( 
		parent->isDescendant( library ) || parent->isDescendant( proto )
		) ) {
		for ( i = origDup.begin(); i != origDup.end(); i++ ) {
			if ( i->first != i->second ) // a global
				i->second->cinfo()->schedule( i->second );
		}
	}

	return child;
}

Element* SimpleElement::copyIntoArray( Element* parent, const string& newName, int n )
		const
{
	return 0;
/*
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
*/
}

#ifdef DO_UNIT_TESTS

static Slot iSlot;
static Slot xSlot;

class CopyTestClass
{
	public:
		CopyTestClass()
			: i_( 0 ), x_( 0.0), s_( "" )
		{;}

		static int getI( const Element* e ) {
			return static_cast< CopyTestClass *>( e->data( 0 ) )->i_;
		}
		static void setI( const Conn* c, int val ) {
			static_cast< CopyTestClass *>( c->data( ) )->i_ = val;
		}
		static double getX( const Element* e ) {
			return static_cast< CopyTestClass *>( e->data( 0 ) )->x_;
		}
		static void setX( const Conn* c, double val ) {
			static_cast< CopyTestClass *>( c->data( ) )->x_ = val;
		}
		static string getS( const Element* e ) {
			return static_cast< CopyTestClass *>( e->data( 0 ) )->s_;
		}
		static void setS( const Conn* c, string val ) {
			static_cast< CopyTestClass *>( c->data( ) )->s_ = val;
		}

		// Sends the x value somewhere.
		static void trigX( const Conn* c ) {
			double x = static_cast< CopyTestClass *>( c->data() )->x_;
			send1< double >( c->target(), xSlot, x );
		}

		// Shared message to request the I value
		static void trigI( const Conn* c ) {
			send0( c->target(), iSlot );
		}

		bool operator==( const CopyTestClass& other )
		{
			return ( i_ == other.i_ && x_ == other.x_ && s_ == other.s_ );
		}
	private:
		int i_;
		double x_;
		string s_;
};

// Gets a vector of elements of c0 and all descendants.
// Only used for tests, so I can be inefficient.
void getCopyTree( Element* c0, vector< Element* >& ret )
{
	ret.push_back( c0 );
	Conn* c = c0->targets( "childSrc" );
	while ( c->good() ) {
		getCopyTree( c->target().e, ret );
		c->increment();
	}
	delete c;
}

// Compares values on two single elements
bool compareCopyValues( const Element* c0, const Element* c1 )
{
	CopyTestClass* v0 = static_cast< CopyTestClass* >( c0->data( 0 ) );
	CopyTestClass* v1 = static_cast< CopyTestClass* >( c1->data( 0 ) );
	
	return ( *v0 == *v1 );
}

// Looks for a match to ic0 among the messages on ic1.
// These are all from the source, so we can look up the msg.
bool checkMsgMatch( Conn* ic0, Conn* ic1, const Element* outsider )
{
	if ( ic0->target().e == outsider ) // No need to try to match.
		return 1;

	while ( ic1->good() ) {
		if ( ic0->target().e->name() == ic1->target().e->name() ||
			( ic0->target().e->name() == "c0" &&
			  ic1->target().e->name() == "c1" )
		)
			if ( ic0->sourceMsg() == ic1->sourceMsg() )
				return 1;
		ic1->increment();
	}
	return 0;
}

// Compares msgs on two single elements, excluding any going to the
// outsider. c0 is original, c1 is dup.
bool compareCopyMsgs( const Element* c0, const Element* c1,
	const Element* outsider )
{
	unsigned int numSrc = c0->cinfo()->numSrc();
	for ( unsigned int i = 0; i < numSrc; i++ ) {

		Conn* ic0 = c0->targets( i );
		while ( ic0->good() )
		{
			Conn* ic1 = c1->targets( i );
			if ( checkMsgMatch( ic0, ic1, outsider ) == 0 )
				return 0;
			delete ic1;
			ic0->increment();
		}
		delete ic0;
	}
	return 1;
}

void copyTest()
{
	cout << "\nTesting copy";

///////////////////////////////////////////////////////////////////////

	static Finfo* iShared[] = {
		new SrcFinfo( "trig", Ftype0::global() ),
		new DestFinfo( "handle", ValueFtype1< double >::global(), 
			RFCAST( &CopyTestClass::setI ) )
	};

	static Finfo* copyFinfos[] = {
		new ValueFinfo( "i", ValueFtype1< int >::global(), 
			GFCAST( &CopyTestClass::getI ),
			RFCAST( &CopyTestClass::setI ) ),
		new ValueFinfo( "x", ValueFtype1< double >::global(), 
			GFCAST( &CopyTestClass::getX ),
			RFCAST( &CopyTestClass::setX ) ),
		new ValueFinfo( "s", ValueFtype1< string >::global(), 
			GFCAST( &CopyTestClass::getS ),
			RFCAST( &CopyTestClass::setS ) ),
		new SharedFinfo( "iShared", iShared, 2 ),
		new SrcFinfo( "xSrc", Ftype1< double >::global() ),
		new DestFinfo( "xDest", Ftype1< double >::global(), 
			RFCAST( &CopyTestClass::setX ) ),
	};

	Cinfo copyTestClass( "CopyClass", "Upi", "Copy Test Class",
				initNeutralCinfo(),
				copyFinfos,
				sizeof( copyFinfos ) / sizeof( Finfo* ),
				ValueFtype1< CopyTestClass >::global() );

	FuncVec::sortFuncVec(); // Only needed here for unit tests

	iSlot = copyTestClass.getSlot( "iShared.trig" );
	xSlot = copyTestClass.getSlot( "xSrc" );

///////////////////////////////////////////////////////////////////////


	Element* n = Neutral::create( "Neutral", "n", Element::root(), Id::scratchId() );
	Element* outsider = Neutral::create( "CopyClass", "outsider", n, Id::scratchId() );
	set< int >( outsider, "i", 0 );
	set< double >( outsider, "x", 0.0 );
	set< string >( outsider, "s", "0.0" );
	ASSERT( outsider != 0, "creating CopyClass" );

	Element* c0 = Neutral::create( "CopyClass", "c0", n, Id::scratchId() );
	set< int >( c0, "i", 10 );
	set< double >( c0, "x", 10.0 );
	set< string >( c0, "s", "10.0" );
	ASSERT( c0 != 0, "creating CopyClass" );

	Element* k0 = Neutral::create( "CopyClass", "k0", c0, Id::scratchId() );
	set< int >( k0, "i", 100 );
	set< double >( k0, "x", 100.0 );
	set< string >( k0, "s", "100.0" );
	ASSERT( k0 != 0, "creating CopyClass child" );

	Element* k1 = Neutral::create( "CopyClass", "k1", c0, Id::scratchId() );
	set< int >( k1, "i", 101 );
	set< double >( k1, "x", 101.0 );
	set< string >( k1, "s", "101.0" );
	ASSERT( k1 != 0, "creating CopyClass child" );

	Element* g1 = Neutral::create( "CopyClass", "g1", k1, Id::scratchId() );
	set< int >( g1, "i", 110 );
	set< double >( g1, "x", 110.0 );
	set< string >( g1, "s", "110.0" );
	ASSERT( g1 != 0, "creating CopyClass grandchild" );

	// Some messages inside tree
	Eref( c0 ).add( "xSrc", k0, "xDest", ConnTainer::Default );
	Eref( k0 ).add( "xSrc", k1, "xDest", ConnTainer::Default );
	Eref( g1 ).add( "xSrc", c0, "xDest", ConnTainer::Default );
	Eref( g1 ).add( "xSrc", k0, "x", ConnTainer::Default );
	Eref( k1 ).add( "iShared", c0, "i", ConnTainer::Default );
	// a couple of messages outside tree
	Eref( k1 ).add( "iShared", outsider, "i", ConnTainer::Default );
	Eref( c0 ).add( "xSrc", outsider, "xDest", ConnTainer::Default );

	/////////////////////////////////////////////////////////////////////
	// Do the copy
	/////////////////////////////////////////////////////////////////////

	Element* c1 = c0->copy( n, "c1" );
	ASSERT( c1 != c0, "copying" );
	ASSERT( c1 != 0, "copying" );

	ASSERT( c1->name() == "c1", "copying" );

	Id p0;
	Id p1;
	get< Id >( c0, "parent", p0 );
	get< Id >( c1, "parent", p1 );
	ASSERT( p0 == n->id(), "copy parent" );
	ASSERT( p1 == n->id(), "copy parent" );

	vector< Id > kids;
	get< vector< Id > >( n, "childList", kids );
	ASSERT( kids.size() == 3 , "copy kids" );
	ASSERT( kids[0] == outsider->id() , "copy kids" );
	ASSERT( kids[1] == c0->id() , "copy kids" );
	ASSERT( kids[2] == c1->id() , "copy kids" );

	vector< Element* > c0family;
	vector< Element* > c1family;
	getCopyTree( c0, c0family );
	getCopyTree( c1, c1family );
	
	ASSERT( c0family.size() == c1family.size(), "copy tree" );
	for ( unsigned int i = 0; i < c0family.size(); i++ ) {
		Element* t0 = c0family[ i ];
		Element* t1 = c1family[ i ];
		ASSERT( t0 != t1, "uniqueness of Elements" );
		ASSERT( t0->id() != t1->id(), "uniqueness of ids" );
		if ( i > 0 )
			ASSERT( t0->name() == t1->name(), "copy names" );
		ASSERT( compareCopyValues( t0, t1 ), "copy values" );
		ASSERT( compareCopyMsgs( t0, t1, outsider ), "copy Msgs" );
	}

	// Check that copy is a unique object
	bool ret;
	int iret;
	ret = set< int >( c1, "i", 333333 );
	ASSERT( ret, "copy uniqueness" );
	get< int >( c0, "i", iret );
	ASSERT( iret == 10, "copy uniqueness" );

	//////////////////////////////////////////////////////////////////
	// Test out copies when there is a global element in the tree.
	// The copy should have messages to the original global element.
	//////////////////////////////////////////////////////////////////

	//////////////////////////////////////////////////////////////////
	// Test out second-generation copies which have a message going to
	// a global element. The messages to the globals should be copied
	// even though the global is not in the tree of the first-generation
	// copies.
	//////////////////////////////////////////////////////////////////

	//////////////////////////////////////////////////////////////////
	// Check copy preserving old name. Copy c0 onto c1.
	//////////////////////////////////////////////////////////////////
	Element* c10 = c0->copy( c1, "" );
	ASSERT( c10 != c0, "copying" );
	ASSERT( c10 != 0, "copying" );
	ASSERT( c10->name() == "c0", "copying" );

	Id p10;
	get< Id >( c10, "parent", p10 );
	ASSERT( p10 == c1->id(), "copy parent" );

	// Check that the copy has a unique id (this was an actual bug!)
	ASSERT( c10->id() != c0->id(), "unique copy id" );

	kids.resize( 0 );
	get< vector< Id > >( n, "childList", kids );
	ASSERT( kids.size() == 3 , "copy kids" );
	ASSERT( kids[0] == outsider->id() , "copy kids" );
	ASSERT( kids[1] == c0->id() , "copy kids" );
	ASSERT( kids[2] == c1->id() , "copy kids" );

	// set( c10, "destroy" );

	set( n, "destroy" );
}

#endif // DO_UNIT_TESTS
