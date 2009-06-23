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
#include "../shell/Shell.h"

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

	Conn* c = targets( "child", 0 ); //zero index for SE
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

Element* SimpleElement::innerCopy( IdGenerator& idGen ) const
{
	SimpleElement* ret = new SimpleElement( this, idGen.next() );

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

Element* SimpleElement::innerCopy( int n, IdGenerator& idGen ) const
{	
	assert( finfo_.size() > 0 );
	assert( dynamic_cast< ThisFinfo* >( finfo_[0] ) != 0 );
	void *data = finfo_[0]->ftype()->copyIntoArray( data_, 1, n );
	ArrayElement* ret =
		new ArrayElement(
			idGen.next(),
			name_,
			cinfo()->numSrc(),
			//msg_,
			//dest_,
			finfo_,
			data,
			n,
			cinfo()->size());
	
	for ( unsigned int i = 1; i < finfo_.size(); i++ ) {
		Finfo* temp = finfo_[i]->copy();
		assert( temp != 0 );
		assert( temp != finfo_[i] );
		ret->addFinfo( temp );
	}
	return ret;
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
 * really do want to make a copy. Otherwise it is not copied,
 * but later clever things are done with the messaging.
 */
Element* SimpleElement::innerDeepCopy(
	map< const Element*, Element* >& tree,
	IdGenerator& idGen ) const
{
	static unsigned int childSrcMsg = 
		initNeutralCinfo()->getSlot( "childSrc" ).msg();

	assert ( childSrcMsg == 0 );

	Element* duplicate = innerCopy( idGen );
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

		// Copy global children only if parent is also global
		if ( !isGlobal() && tgt->isGlobal() ) 
			continue;

		if ( tree.find( tgt ) != tree.end() )
			cout << "Warning: SimpleElement::innerDeepCopy: Loop in element tree at " << tgt->name() << endl;
		else 
			tgt->innerDeepCopy( tree, idGen );
	}
	return duplicate;
}

Element* SimpleElement::innerDeepCopy(
	map< const Element*, Element* >& tree,
	int n,
	IdGenerator& idGen ) const
{
	static unsigned int childSrcMsg = 
		initNeutralCinfo()->getSlot( "childSrc" ).msg();

	assert ( childSrcMsg == 0 );

	Element* duplicate = innerCopy( n, idGen );
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

		// Copy global children only if parent is also global
		if ( !isGlobal() && tgt->isGlobal() ) 
			continue;

		if ( tree.find( tgt ) != tree.end() )
			cout << "Warning: SimpleElement::innerDeepCopy: Loop in element tree at " << tgt->name() << endl;
		else 
			tgt->innerDeepCopy( tree, n, idGen );
	}
	return duplicate;
}

/**
 * Copies messages between current element and a global, to the
 * duplicate element and the same global. Does so both for src and dest
 * messages.
 */
void SimpleElement::copyGlobalMessages( Element* dup, bool isArray ) const
{

	unsigned int numSrc = cinfo()->numSrc();
	for ( unsigned int i = 1; i < numSrc; i++ ) { // skip childSrc msg.
		const Msg* m = &( msg_[i] );
		while ( m ) {
			if ( m->size() > 0 ) {
				if ( !m->isDest() ) {
					vector< ConnTainer* >::const_iterator c;
					for ( c = m->begin(); c != m->end(); c++ ) {
						Element* tgt = ( *c )->e2();
						if ( tgt == this )
							tgt = ( *c )->e1();
						if ( tgt->isGlobal() ) {
							m->copy( *c, dup, tgt, isArray );
						}
					}
				}
			}
			m = m->next( this );
		}
	}
	
	/*
	vector< Msg >::const_iterator m;
	if ( msg_.size() > 1 ) { 
		// Begin at msg + 1 because we don't want to copy childSrc msgs.
		for ( m = msg_.begin() + 1; m != msg_.end(); m++ ) {
			if ( m->size() == 0 )
				continue;
			vector< ConnTainer* >::const_iterator c;
			for ( c = m->begin(); c != m->end(); c++ ) {
				Element* tgt = ( *c )->e2();
				if ( tgt == this )
					tgt = ( *c )->e1();
				if ( tgt->isGlobal() ) {
					m->copy( *c, dup, tgt, isArray );
				}
			}
		}
	}
	*/

	// Now we iterate through dests.
	// Perhaps should check for child msgs?
	map< int, vector< ConnTainer* > >::const_iterator k;
	for ( k = dest_.begin(); k != dest_.end(); ++k ) {
		vector< ConnTainer* >::const_iterator c;
		for ( c = k->second.begin(); c != k->second.end(); c++ ) {
			// Here check orientation of e1 and msg2
			Element* tgt = ( *c )->e1();
			assert( tgt != this );
			if ( tgt->isGlobal() ) {
				assert( ( *c )->msg1() >= 0 );
				unsigned int msgNum = ( *c )->msg1();
				assert( msgNum < tgt->numMsg() );
				const Msg* m = tgt->msg( msgNum );
				m->copy( *c, tgt, dup, isArray );
			}
		}
	}
}

/**
 * Copies messages from current element to duplicate provided dest is
 * also on tree.
 */
void SimpleElement::copyMessages( Element* dup, 
	map< const Element*, Element* >& origDup, bool isArray ) const
{
	
	map< const Element*, Element* >::iterator k;
	/*
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
				m->copy( *c, dup, k->second, isArray );
			}
		}
	}
	*/

	unsigned int numSrc = cinfo()->numSrc();
	for ( unsigned int i = 0; i < numSrc; i++ ) {
		const Msg* m = &( msg_[i] );
		while ( m ) {
			if ( m->size() > 0 ) {
				if ( !m->isDest() ) {
					vector< ConnTainer* >::const_iterator c;
						for ( c = m->begin(); c != m->end(); c++ ) {
							k = origDup.find( ( *c )->e2() );
							// cout << "copyMessages: srcMsg=" << i << ", msg[i],m =" << &( msg_[i] ) << ", " << m << ", funcids=" << msg_[i].funcId() << "," << m->funcId() << ", c->msg1,2=" << ( *c )->msg1() << ", " << ( *c )->msg2() << endl;
							if ( k != origDup.end() ) {
								m->copy( *c, dup, k->second, isArray );
						}
					}
				}
			}
			m = m->next( this );
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
 * If an Id is given, it is assigned to the newly created copy.
 * Otherwise (if the last argument is skipped), an id is generated locally from
 * the block of Ids allocated to this node.
 * It is non-recursive but calls lots recursive functions.
 *
 * A special case happens if one of the source elements in the tree
 * has the isGlobal flag. This flag means that the source, typically
 * an object on the library, is not to be duplicated but instead
 * its messages into the tree should be duplicated. This is used,
 * for example, for HHGates on HHChannels. All channel duplicates
 * should use the same HHGates.
 * A further refinement is that global elements can be copied but only
 * if they are the root of the copy tree. In this case we assume that
 * the user really does want to copy the global element as they have
 * specifically requested the copy.
 *
 * \todo: This will need a lot of work to handle cross-node copies,
 * or even worse, copies of element trees that span nodes. For now
 * it is single-node stuff.
 */
Element* SimpleElement::copy(
	Element* parent,
	const string& newName,
	IdGenerator& idGen ) const
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
	// However, if orig is a Global, copy is not done, unless this
	// originating element is itself a global.
	// Note that here we use Global in a different sense than in the
	// objects that are children of /library and are therefore common on
	// all nodes. Here Global just means that they should not be copied.
	// I don't think this situation really comes up.
	map< const Element*, Element* > origDup;
	map< const Element*, Element* >::iterator i;

	// vector< pair< Element*, unsigned int > > delConns;

	Element* child = innerDeepCopy( origDup, idGen );
	child->setName( nm );

	/*
	 * Sanity check
	for ( i = origDup.begin(); i != origDup.end(); i++ ) {
		if ( i->first->isGlobal() )
			cout << "origDup has a global " << i->first->name() << endl;
		if ( i->first->isGlobal() || i->second->isGlobal() )
			cout << "origDup has a global " << i->second->name() << endl;
	}
	*/

	// Phase 2. Copy over messages that are within the tree.
	// Here we need only copy from message sources.
	for ( i = origDup.begin(); i != origDup.end(); i++ ) {
		i->first->copyMessages( i->second, origDup, false );
	}
	
	// Phase 3 : Copy over messages to any global elements
	// Here we have to deal with message sources as well as dests.
	for ( i = origDup.begin(); i != origDup.end(); i++ ) {
		i->first->copyGlobalMessages( i->second, false );
	}
	
	// Phase 4: stick the copied tree onto the parent Element.
	ret = Eref( parent ).add( "childSrc", child, "child", 
		ConnTainer::Default );
	assert( ret );

	// Phase 5: Schedule all the objects
	if ( !( 
		parent->isDescendant( library ) || parent->isDescendant( proto )
	) ) {
		for ( i = origDup.begin(); i != origDup.end(); i++ ) {
			if ( i->first != i->second ) // a global
				i->second->cinfo()->schedule( i->second,
					ConnTainer::Default );
		}
	}

	// Phase 6: Set to globals or local node, as the case may be.
	unsigned int node = Shell::myNode();
	if ( parent != Element::root() && parent->id().isGlobal() )
		node = Id::GlobalNode;
	for ( i = origDup.begin(); i != origDup.end(); i++ ) {
		if ( i->first != i->second )
			i->second->id().setNode( node );
	}

	return child;
}

Element* SimpleElement::copyIntoArray(
	Id parent,
	const string& newName,
	int n,
	IdGenerator& idGen ) const
{
	static const Element* library = Id( "/library" )();
	static const Element* proto = Id( "/proto" )();

	if ( parent()->isDescendant( this ) ) {
		cout << "Warning: SimpleElement::copyIntoArray: Attempt to copy within descendant tree" << parent()->name() << endl;
		return 0;
	}
	string nm = newName;

	if ( newName == "" )
		nm = name();
	Id oldChild;
	//oldChild = Neutral::getChildByName( parent, nm)
	bool ret = lookupGet< Id, string >(
					parent.eref(), "lookupChild", oldChild, nm );
	assert( ret );
	if ( !oldChild.bad() ) {
		cout << "Warning: SimpleElement::copy: pre-existing child with target name: " << parent()->name() << "/" << nm << endl;
		return 0;
	}
	
	// Phase 1. Copy Elements, but not building up parent-child info.
	// First is original, second is copy
	// However, if it was a Global, both original and second are the same.
	map< const Element*, Element* > origDup;
	map< const Element*, Element* >::iterator i;

	// vector< pair< Element*, unsigned int > > delConns;

	Element* child = innerDeepCopy( origDup, n, idGen );
	child->setName( nm );
	
	
	
	// Phase 2. Copy over messages that are within the tree.
	// Here we need only copy from message sources.
	for ( i = origDup.begin(); i != origDup.end(); i++ ) {
		i->first->copyMessages( i->second, origDup, true );
	}
	
	// vector <Id> kids;
	// get< vector< Id > >( Eref(child, 1), "childList", kids );
	
	
	// Phase 3 : Copy over messages to any global elements
	// Here we have to deal with message sources as well as dests.
	for ( i = origDup.begin(); i != origDup.end(); i++ ) {
		i->first->copyGlobalMessages( i->second, true );
	}
	
	
	// Phase 4: stick the copied tree onto the parent Element.
	ret = parent.eref().add( "childSrc", child, "child", 
		ConnTainer::One2All );
	assert( ret );

	// Phase 5: Schedule all the objects
	if ( !( 
		parent()->isDescendant( library ) || parent()->isDescendant( proto )
		) ) {
		for ( i = origDup.begin(); i != origDup.end(); i++ ) {
			if ( i->first != i->second ) // a global
				i->second->cinfo()->schedule( i->second, ConnTainer::One2All );
		}
	}
	return child;
}

#ifdef DO_UNIT_TESTS
#include "DeletionMarkerFinfo.h"
#include "GlobalMarkerFinfo.h"

static Slot iSlot;
static Slot xSlot;
static Slot xDestSlot;

class CopyTestClass
{
	public:
		CopyTestClass()
			: i_( 0 ), x_( 0.0), s_( "" )
		{;}
		
		virtual ~CopyTestClass(){;}
		
		static int getI( Eref e ) {
			return static_cast< CopyTestClass *>( e.data(  ) )->i_;
		}
		static void setI( const Conn* c, int val ) {
			static_cast< CopyTestClass *>( c->data( ) )->i_ = val;
		}
		static double getX( Eref e ) {
			return static_cast< CopyTestClass *>( e.data(  ) )->x_;
		}
		static void setX( const Conn* c, double val ) {
			static_cast< CopyTestClass *>( c->data( ) )->x_ = val;
		}
		static string getS( Eref e ) {
			return static_cast< CopyTestClass *>( e.data(  ) )->s_;
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
void getCopyTree( Eref c0, vector< Eref >& ret )
{
	ret.push_back( c0 );
	Conn* c = c0.e->targets( "childSrc", c0.i); // zero index for SE
	while ( c->good() ) {
		getCopyTree( c->target(), ret );
		c->increment();
	}
	delete c;
}

// Compares values on two single elements
bool compareCopyValues( const Eref c0, const Eref c1 )
{
	CopyTestClass* v0 = static_cast< CopyTestClass* >( c0.e->data( c0.i ) );
	CopyTestClass* v1 = static_cast< CopyTestClass* >( c1.e->data( c1.i ) );
	
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
			  ic1->target().e->name() == "c1" ) ||
			( ic0->target().e->name() == "c0" &&
			  ic1->target().e->name() == "cc" )  
		)
			if ( ic0->sourceMsg() == ic1->sourceMsg() )
				return 1;
		ic1->increment();
	}
	return 0;
}

// Compares msgs on two single elements, excluding any going to the
// outsider. c0 is original, c1 is dup.
bool compareCopyMsgs( const Eref c0, const Eref c1,
	const Element* outsider )
{	
	unsigned int numSrc = c0.e->cinfo()->numSrc();
	for ( unsigned int i = 0; i < numSrc; i++ ) {

		Conn* ic0 = c0.e->targets( i, c0.i );
		while ( ic0->good() )
		{
			Conn* ic1 = c1.e->targets( i, c1.i );
			if ( checkMsgMatch( ic0, ic1, outsider ) == 0 )
				return 0;
			delete ic1;
			ic0->increment();
		}
		delete ic0;
	}
	return 1;
}

Element* makeCopyTree( Element* n, Element* outsider )
{
	Element* c0 = Neutral::create( "CopyClass", "c0", n->id(), Id::scratchId() );
	set< int >( c0, "i", 10 );
	set< double >( c0, "x", 10.0 );
	set< string >( c0, "s", "10.0" );
	ASSERT( c0 != 0, "creating CopyClass" );

	Element* k0 = Neutral::create( "CopyClass", "k0", c0->id(), Id::scratchId() );
	set< int >( k0, "i", 100 );
	set< double >( k0, "x", 100.0 );
	set< string >( k0, "s", "100.0" );
	ASSERT( k0 != 0, "creating CopyClass child" );

	Element* k1 = Neutral::create( "CopyClass", "k1", c0->id(), Id::scratchId() );
	set< int >( k1, "i", 101 );
	set< double >( k1, "x", 101.0 );
	set< string >( k1, "s", "101.0" );
	ASSERT( k1 != 0, "creating CopyClass child" );

	Element* g1 = Neutral::create( "CopyClass", "g1", k1->id(), Id::scratchId() );
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

	return c0;
}

Element* checkBasicCopy( Element* c0, Element* n, Element* outsider,
	vector< Eref >& c0family )
{
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

	vector< Eref > c1family;
	getCopyTree( Eref(c0, 0), c0family );
	getCopyTree( Eref(c1, 0), c1family );
	
	ASSERT( c0family.size() == c1family.size(), "copy tree" );
	for ( unsigned int i = 0; i < c0family.size(); i++ ) {
		Eref t0 = c0family[ i ];
		Eref t1 = c1family[ i ];
		ASSERT( t0.e != t1.e, "uniqueness of Elements" );
		ASSERT( t0.id() != t1.id(), "uniqueness of ids" );
		if ( i > 0 ) {
			ASSERT( t0.name() == t1.name(), "copy names" );
		}
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

	return c1;
}

Element* check1stGenGlobalCopy( Element* n, Element* c1 )
{
	vector< Id > kids;
	get< vector< Id > >( c1, "childList", kids );
	ASSERT( kids.size() == 2 , "copy kids" );
	Element* k0 = kids[0]();
	ASSERT( k0->name() == "k0" , "copy global kids" );


	// Correct numbers of targets before the copy.
	ASSERT( k0->numTargets( xSlot.msg() ) == 1, "copying globals" );
	ASSERT( k0->numTargets( xDestSlot.msg() ) == 1, "copying globals" );

	k0->addFinfo( GlobalMarkerFinfo::global() ); // Mark as a global.
	Element* c2 = c1->copy( n, "c2" );

	ASSERT( c2 != c1, "copying Globals" );
	ASSERT( c2 != 0, "copying Globals" );
	ASSERT( c2->name() == "c2", "copying Globals" );

	// Now check for messages. We expect to use the old k0 as a
	// xSrc to k1 and as a xDest from c2.xSrc.
	// k1 is a sibling of k0, and g1 a child of k1. We do not expect
	// a copy of k0, but we do of the other two.
	// Eref( k0 ).add( "xSrc", k1, "xDest", ConnTainer::Default );
	// Eref( c0 ).add( "xSrc", k0, "xDest", ConnTainer::Default );
	
	Element* k1 = kids[1]();

	// Correct kids are copied
	kids.resize( 0 );
	get< vector< Id > >( c2, "childList", kids );
	ASSERT( kids.size() == 1 , "copying globals" );
	Element* newK1 = kids[0]();
	ASSERT( newK1->name() == "k1", "copying globals" );

	Id newG1Id;
	lookupGet< Id, string >( newK1, "lookupChild", newG1Id, "g1" );
	Element* newG1 = newG1Id();
	ASSERT( newG1->name() == "g1", "copying globals" );

	// Correct numbers of targets following the copy.
	ASSERT( k0->numTargets( xSlot.msg() ) == 2, "copying globals" );
	ASSERT( k0->numTargets( xDestSlot.msg() ) == 2, "copying globals" );

	// Both c2 and c1 point toward k0.
	vector< ConnTainer* >::const_iterator i = 
		c2->msg( xSlot.msg() )->begin();
	ASSERT( (*i)->e1() == c2, "Copying globals" );
	ASSERT( (*i)->e2() == k0, "Copying globals" );

	i = c1->msg( xSlot.msg() )->begin();
	ASSERT( (*i)->e1() == c1, "Copying globals" );
	ASSERT( (*i)->e2() == k0, "Copying globals" );

	// Both k1 and newK1 get msgs from k0
	i = k0->msg( xSlot.msg() )->begin();
	ASSERT( (*i)->e1() == k0, "Copying globals" );
	ASSERT( (*i)->e2() == k1, "Copying globals" );
	i++;
	ASSERT( (*i)->e1() == k0, "Copying globals" );
	ASSERT( (*i)->e2() == newK1, "Copying globals" );

	// Now turn it around. k0 gets msgs from c1 and c2, in that order.
	const vector< ConnTainer* >* dest = k0->dest( xDestSlot.msg() );
	ASSERT( dest->size() == 2, "COPYING globals" );
	ASSERT( (*dest)[0]->e1() == c1, "Copying globals" );
	ASSERT( (*dest)[0]->e2() == k0, "Copying globals" );
	ASSERT( (*dest)[1]->e1() == c2, "Copying globals" );
	ASSERT( (*dest)[1]->e2() == k0, "Copying globals" );

	// k1 gets a single message from k0
	dest = k1->dest( xDestSlot.msg() );
	ASSERT( dest->size() == 1, "COPYING globals" );
	ASSERT( (*dest)[0]->e1() == k0, "Copying globals" );
	ASSERT( (*dest)[0]->e2() == k1, "Copying globals" );

	// newK1 gets a single message from k0
	dest = newK1->dest( xDestSlot.msg() );
	ASSERT( dest->size() == 1, "COPYING globals" );
	ASSERT( (*dest)[0]->e1() == k0, "Copying globals" );
	ASSERT( (*dest)[0]->e2() == newK1, "Copying globals" );

	return c2;
}

// This is much the same as above. Assumes that the above copies are done.
void check2ndGenGlobalCopy( Element* n, Element* c1, Element* c2 )
{

	vector< Id > kids;
	get< vector< Id > >( c1, "childList", kids );
	ASSERT( kids.size() == 2 , "copy kids" );
	Element* k0 = kids[0]();
	ASSERT( k0->name() == "k0" , "copy global kids" );

	// Correct numbers of targets before the copy.
	ASSERT( k0->numTargets( xSlot.msg() ) == 2, "copying globals" );
	ASSERT( k0->numTargets( xDestSlot.msg() ) == 2, "copying globals" );

	Element* c3 = c2->copy( n, "c3" );

	ASSERT( c3 != c2, "copying Globals" );
	ASSERT( c3 != 0, "copying Globals" );
	ASSERT( c3->name() == "c3", "copying Globals" );

	// Now check for messages. We expect to use the old k0 as a
	// xSrc to k1 and as a xDest from c3.xSrc.
	// k1 is a sibling of k0, and g1 a child of k1. We do not expect
	// a copy of k0, but we do of the other two.
	// Eref( k0 ).add( "xSrc", k1, "xDest", ConnTainer::Default );
	// Eref( c0 ).add( "xSrc", k0, "xDest", ConnTainer::Default );
	
	Element* k1 = kids[1]();

	// Correct kids are copied
	kids.resize( 0 );
	get< vector< Id > >( c3, "childList", kids );
	ASSERT( kids.size() == 1 , "copying globals" );
	Element* newK1 = kids[0]();
	ASSERT( newK1->name() == "k1", "copying globals" );

	Id newG1Id;
	lookupGet< Id, string >( newK1, "lookupChild", newG1Id, "g1" );
	Element* newG1 = newG1Id();
	ASSERT( newG1->name() == "g1", "copying globals" );

	// Correct numbers of targets following the copy.
	ASSERT( k0->numTargets( xSlot.msg() ) == 3, "copying globals" );
	ASSERT( k0->numTargets( xDestSlot.msg() ) == 3, "copying globals" );

	// all of  c3, c2 and c1 point toward k0.
	vector< ConnTainer* >::const_iterator i = 
		c3->msg( xSlot.msg() )->begin();
	ASSERT( (*i)->e1() == c3, "Copying globals" );
	ASSERT( (*i)->e2() == k0, "Copying globals" );

	i = c2->msg( xSlot.msg() )->begin();
	ASSERT( (*i)->e1() == c2, "Copying globals" );
	ASSERT( (*i)->e2() == k0, "Copying globals" );

	i = c1->msg( xSlot.msg() )->begin();
	ASSERT( (*i)->e1() == c1, "Copying globals" );
	ASSERT( (*i)->e2() == k0, "Copying globals" );

	// Both k1 and newK1 get msgs from k0
	i = k0->msg( xSlot.msg() )->begin();
	ASSERT( (*i)->e1() == k0, "Copying globals" );
	ASSERT( (*i)->e2() == k1, "Copying globals" );
	i++; // This would bring us to c2/k1
	i++; // Here we are now with c3/k1
	ASSERT( (*i)->e1() == k0, "Copying globals" );
	ASSERT( (*i)->e2() == newK1, "Copying globals" );

	// Now turn it around. k0 gets msgs from c1, c2 and c3, in that order.
	const vector< ConnTainer* >* dest = k0->dest( xDestSlot.msg() );
	ASSERT( dest->size() == 3, "COPYING globals" );
	ASSERT( (*dest)[0]->e1() == c1, "Copying globals" );
	ASSERT( (*dest)[0]->e2() == k0, "Copying globals" );
	ASSERT( (*dest)[1]->e1() == c2, "Copying globals" );
	ASSERT( (*dest)[1]->e2() == k0, "Copying globals" );
	ASSERT( (*dest)[2]->e1() == c3, "Copying globals" );
	ASSERT( (*dest)[2]->e2() == k0, "Copying globals" );

	// k1 gets a single message from k0
	dest = k1->dest( xDestSlot.msg() );
	ASSERT( dest->size() == 1, "COPYING globals" );
	ASSERT( (*dest)[0]->e1() == k0, "Copying globals" );
	ASSERT( (*dest)[0]->e2() == k1, "Copying globals" );

	// newK1 gets a single message from k0
	dest = newK1->dest( xDestSlot.msg() );
	ASSERT( dest->size() == 1, "COPYING globals" );
	ASSERT( (*dest)[0]->e1() == k0, "Copying globals" );
	ASSERT( (*dest)[0]->e2() == newK1, "Copying globals" );
}

///////////////////////////////////////////////////////////////////////
// This is the wrapper test function for copies.
///////////////////////////////////////////////////////////////////////

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
	xDestSlot = copyTestClass.getSlot( "xDest" );

///////////////////////////////////////////////////////////////////////


	Element* n = Neutral::create( "Neutral", "n", Element::root()->id(), Id::scratchId() );
	Element* outsider = Neutral::create( "CopyClass", "outsider", n->id(), Id::scratchId() );
	set< int >( outsider, "i", 0 );
	set< double >( outsider, "x", 0.0 );
	set< string >( outsider, "s", "0.0" );
	ASSERT( outsider != 0, "creating CopyClass" );

	Element* c0 = makeCopyTree( n, outsider );

	/////////////////////////////////////////////////////////////////////
	// Do and check the copy
	/////////////////////////////////////////////////////////////////////
	
	vector< Eref > c0family;
	Element* c1 = checkBasicCopy( c0, n, outsider, c0family );

	//////////////////////////////////////////////////////////////////
	// Test out copies when there is a global element in the tree.
	// The copy should have messages to the original global element.
	//////////////////////////////////////////////////////////////////
	
	Element* c2 = check1stGenGlobalCopy( n, c1 );

	//////////////////////////////////////////////////////////////////
	// Test out second-generation copies which have a message going to
	// a global element. The messages to the globals should be copied
	// even though the global is not in the tree of the first-generation
	// copies.
	//////////////////////////////////////////////////////////////////
	
	check2ndGenGlobalCopy( n, c1, c2 );

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

	vector< Id > kids;
	get< vector< Id > >( n, "childList", kids );
	ASSERT( kids.size() == 5 , "copy kids" );
	ASSERT( kids[0] == outsider->id() , "copy kids" );
	ASSERT( kids[1] == c0->id() , "copy kids" );
	ASSERT( kids[2] == c1->id() , "copy kids" );

	// set( c10, "destroy" );
	
// 	create Neutral m 
// 	create CopyClass m/c_simple
// 	createmap m/c_s

	Element* cc = c0->copyIntoArray( n->id(), "cc", 10 );
	ASSERT( cc != c0, "copying" );
	ASSERT( cc != 0, "copying" );

	ASSERT( cc->name() == "cc", "copying" );

	Id p0;
	Id p1;
	get< Id >( c0, "parent", p0 );
	get< Id >( cc, "parent", p1 );
	ASSERT( p0 == n->id(), "copy parent" );
	ASSERT( p1 == n->id(), "copy parent" );

	get< vector< Id > >( n, "childList", kids );
	ASSERT( kids.size() == 15 , "copy kids" ); // what should we do about it? 
	ASSERT( kids[0] == outsider->id() , "copy kids" );
	ASSERT( kids[1] == c0->id() , "copy kids" );
	ASSERT( kids[2] == c1->id() , "copy kids" );
	ASSERT( kids[5] == cc->id().assignIndex(0) , "copy kids" );
	
	vector< Eref > ccfamily;
	getCopyTree( Eref(cc, 0), ccfamily );
	ASSERT( c0family.size() == ccfamily.size(), "copy tree" );
	for ( unsigned int i = 0; i < c0family.size(); i++ ) {
		Eref t0 = c0family[ i ];
		Eref t1 = ccfamily[ i ];
		ASSERT( t0.e != t1.e, "uniqueness of Elements" );
		ASSERT( t0.id() != t1.id(), "uniqueness of ids" );
		if ( i > 0 ) {
			ASSERT( t0.e->name() == t1.e->name(), "copy names" );
		}
		ASSERT( compareCopyValues( t0, t1 ), "copy values" );
//		ASSERT( compareCopyMsgs( t0, t1, outsider ), "copy Msgs" );
	}

	Element* cc0 = c0->copyIntoArray( cc->id().assignIndex(2), "", 10 );
	ASSERT( cc0 != c0, "copying" );
	ASSERT( cc0 != 0, "copying" );
	ASSERT( cc0->name() == "c0", "copying" );

	get< Id >( cc0, "parent", p10 );
	ASSERT( p10 == cc->id().assignIndex(2), "copy parent" );

	// Check that the copy has a unique id (this was an actual bug!)
	ASSERT( cc0->id() != c0->id(), "unique copy id" );

	kids.resize( 0 );
	get< vector< Id > >( n, "childList", kids );
	ASSERT( kids.size() == 15 , "copy kids" );
	ASSERT( kids[0] == outsider->id() , "copy kids" );
	ASSERT( kids[1] == c0->id() , "copy kids" );
	ASSERT( kids[2] == c1->id() , "copy kids" );
	ASSERT( kids[5] == cc->id().assignIndex(0) , "copy kids" );
	
	Element* m = Neutral::create( "Neutral", "m", Element::root()->id(), Id::scratchId() );
	Element *c_simple = Neutral::create( "CopyClass", "c_simple", m->id(), Id::scratchId() );
	bool ret = set <double> (c_simple, "x", 100);
	ASSERT(ret, "set value to compartment");
	Element *c_array = c_simple->copyIntoArray(m->id(), "c_array", 4);
	ASSERT(c_array->numEntries() == 4, "number of entries");
	ASSERT(c_array != 0, "simple element copied into array element");
	Eref eref = Eref(c_array, 2);
	double x;
	get <double> (eref, "x", x);
	ASSERT(x == 100, "checking initial value of index element");
	set <double> (eref, "x", 200);
	get <double> (eref, "x", x);
	ASSERT(x == 200, "checking index element");
	eref = Eref(c_array, 1);
	get <double> (eref, "x", x);
	get <double> (c_simple, "x", x);
	ASSERT(x == 100, "checking other index element");
	set( m, "destroy" );
	set( n, "destroy" );
}

#endif // DO_UNIT_TESTS
