/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
#include <map>
#include "Cinfo.h"
#include "MsgSrc.h"
#include "MsgDest.h"
#include "SimpleElement.h"
#include "send.h"
#include "DynamicFinfo.h"
#include "ValueFinfo.h"
#include "DerivedFtype.h"
#include "Ftype2.h"
#include "ValueFtype.h"
#include "SrcFinfo.h"
#include "DestFinfo.h"
#include "SharedFtype.h"
#include "SharedFinfo.h"
#include "LookupFinfo.h"
#include "LookupFtype.h"

#include "Neutral.h"

static Finfo* neutralFinfos[] = 
{
		new ValueFinfo( "name", ValueFtype1< string >::global(),
					reinterpret_cast< GetFunc >( &Neutral::getName ),
					reinterpret_cast< RecvFunc >( &Neutral::setName )
		),
		new ValueFinfo( "parent", ValueFtype1< unsigned int >::global(),
					reinterpret_cast< GetFunc >( &Neutral::getParent ),
					&dummyFunc
		),
		new ValueFinfo( "childList",
				ValueFtype1< vector< unsigned int > >::global(), 
				reinterpret_cast< GetFunc>( &Neutral::getChildList ),
				&dummyFunc
		),
		new LookupFinfo(
				"lookupChild",
				LookupFtype< int, string >::global(), 
				reinterpret_cast< GetFunc >( &Neutral::getChildByName ),
				0
		),
		new SrcFinfo( "childSrc", Ftype1< int >::global() ),
		new DestFinfo( "child", Ftype1< int >::global(),
			reinterpret_cast< RecvFunc >( &Neutral::childFunc ) ),
		new DestFinfo( "create", Ftype2< string, string >::global(),
			reinterpret_cast< RecvFunc >( &Neutral::create ) ),
		new DestFinfo( "destroy", Ftype0::global(),
			&Neutral::destroy ),
};

static Cinfo neutralCinfo(
				"Neutral",
				"Upi Bhalla",
				"Neutral object. Manages Element heirarchy.",
				"",
				neutralFinfos,
				sizeof( neutralFinfos ) / sizeof( Finfo* ),
				ValueFtype1< Neutral >::global()
);

/**
 * Declaration of the Element::root() function is here because
 * we want to be able to set it up as a Neutral. This function
 * uses the common trick of having an internal static value which
 * is created the first time the function is called.
 * This is an unusual Element because it is created on every node.
 */
Element* Element::root()
{
	// elementList.reserve( 128 );
	static Element* ret = neutralCinfo.create( "root" );
	
	return ret;
}

//////////////////////////////////////////////////////////////////
// Here we put the Neutral class functions.
//////////////////////////////////////////////////////////////////

/**
 * This function is called to recursively delete all children
 * It is a bit tricky, because while we delete things the conn
 * vector gets altered as each child is removed. So the iterators
 * don't work.
 * \todo It actually needs to work in three stages.
 * 1. Mark all children for deletion.
 * 2. Clear out messages outside local set, without altering
 * local Conn arrays.
 * 3. Delete.
 */
void Neutral::childFunc( const Conn& c , int stage )
{
		Element* e = c.targetElement();
		assert( stage == 0 || stage == 1 || stage == 2 );

		switch ( stage ) {
				case MARK_FOR_DELETION:
					send1< int >( e, 0, MARK_FOR_DELETION );
					e->prepareForDeletion( 0 );
				break;
				case CLEAR_MESSAGES:
					send1< int >( e, 0, CLEAR_MESSAGES );
					e->prepareForDeletion( 1 );
				break;
				case COMPLETE_DELETION:
					send1< int >( e, 0, COMPLETE_DELETION );
					delete e;
				break;
				default:
					assert( 0 );
				break;
		}
}

const string Neutral::getName( const Element* e )
{
		return e->name();
}

void Neutral::setName( const Conn& c, const string s )
{
	c.targetElement()->setName( s );
}

// Perhaps this should take a Cinfo* for the first arg, except that
// I don't want to add yet another class into the header.
// An alternative would be to use an indexed lookup for all Cinfos
void Neutral::create( const Conn& conn,
				const string cinfo, const string name )
{
		Element* e = conn.targetElement();

		// Need to check here if the name is an existing one.
		const Cinfo* c = Cinfo::find( cinfo );
		if ( c ) {
			Element* kid = c->create( name );
			// Here a global absolute or a relative finfo lookup for
			// the childSrc field would be useful.
			e->findFinfo( "childSrc" )->
					add( e, kid, kid->findFinfo( "child" ) ); 
		} else {
			cout << "Error: Neutral::create: class " << cinfo << 
					" not found\n";
		}
}

void Neutral::destroy( const Conn& c )
{
	childFunc( c, MARK_FOR_DELETION );
	childFunc( c, CLEAR_MESSAGES );
	childFunc( c, COMPLETE_DELETION );
}

unsigned int Neutral::getParent( const Element* e )
{
	const SimpleElement* se = dynamic_cast< const SimpleElement* >( e );
	assert( se != 0 );
	assert( se->destSize() > 0 );
	// The zero dest is the child dest.
	assert( se->connDestEnd( 0 ) > se->connDestBegin( 0 ) );
	// return reverseElementLookup( se->connDestBegin()->targetElement() );
	return se->connDestBegin( 0 )->targetElement()->id();
}

/**
 * Looks up the child with the specified name, and returns the eid.
 */
int Neutral::getChildByName( const Element* elm, const string s )
{
	const SimpleElement* e = dynamic_cast< const SimpleElement *>(elm);
	assert( e != 0 );
	// assert that the element is a neutral.

	// Here we should put in one of the STL algorithms.
	vector< Conn >::const_iterator i;
	// For neutral, src # 0 is the childSrc.
	vector< Conn >::const_iterator begin = e->connSrcBegin( 0 );
	vector< Conn >::const_iterator end = e->connSrcEnd( 0 );
	for ( i = begin; i != end; i++ ) {
		if ( i->targetElement()->name() == s ) {
			return i->targetElement()->id();
		}
	}
	// Failure option: return root id.
	return 0;
}

/**
 * Looks up the child with the specified name, and sends its eid
 * in a message back to sender.
 */
void Neutral::lookupChild( const Conn& c, const string s )
{
	SimpleElement* e =
			dynamic_cast< SimpleElement* >( c.targetElement() );
	assert( e != 0 );
	// assert that the element is a neutral.

	// Here we should put in one of the STL algorithms.
	vector< Conn >::const_iterator i;
	// For neutral, src # 0 is the childSrc.
	vector< Conn >::const_iterator begin = e->connSrcBegin( 0 );
	vector< Conn >::const_iterator end = e->connSrcEnd( 0 );
	for ( i = begin; i != end; i++ ) {
		if ( i->targetElement()->name() == s ) {
			// For neutral, src # 1 is the shared message.
			sendTo1< unsigned int >( e, 1, c.sourceIndex( e ), 
				i->targetElement()->id() );
			return;
		}
	}
	// Hm. What is the best thing to do if it fails? Return an
	// error value, or not return anything at all?
	// Perhaps best to be consistent about returning something.
	sendTo1< unsigned int >( e, 1, c.sourceIndex( e ), MAXUINT );
}

vector< unsigned int > Neutral::getChildList( const Element* elm )
{
	const SimpleElement* e = dynamic_cast< const SimpleElement *>(elm);
	assert( e != 0 );

	vector< Conn >::const_iterator i;
	// For neutral, src # 0 is the childSrc.
	vector< Conn >::const_iterator begin = e->connSrcBegin( 0 );
	vector< Conn >::const_iterator end = e->connSrcEnd( 0 );

	vector< unsigned int > ret;
	if ( end == begin ) // zero children
			return ret;
	ret.reserve( end - begin );
	for ( i = begin; i != end; i++ )
		ret.push_back( i->targetElement()->id() );

	return ret;
}

/////////////////////////////////////////////////////////////////////

#ifdef DO_UNIT_TESTS
#include "Ftype2.h"
#include "setget.h"

void testNeutral()
{
		cout << "\nTesting Neutral";

		Element* n1 = neutralCinfo.create( "n1" );
		string s;
		get< string >( n1, n1->findFinfo( "name" ), s );
		ASSERT( s == "n1", "Neutral name get" );
		set< string >( n1, n1->findFinfo( "name" ), "N1" );
		s = "";
		get< string >( n1, n1->findFinfo( "name" ), s );
		ASSERT( s == "N1", "Neutral name set" );

		Element* n2 = neutralCinfo.create( "n2" );
		
		ASSERT( n1->findFinfo( "childSrc" )->add(
								n1, n2, n2->findFinfo( "child" ) ),
						"adding child"
			  );

		Element* n3 = neutralCinfo.create( "n3" );
		
		ASSERT( n1->findFinfo( "childSrc" )->add(
								n1, n3, n3->findFinfo( "child" ) ),
						"adding child"
			  );

		Element* n21 = neutralCinfo.create( "n21" );
		
		ASSERT( n2->findFinfo( "childSrc" )->add(
								n2, n21, n21->findFinfo( "child" ) ),
						"adding child"
			  );

		Element* n22 = neutralCinfo.create( "n22" );
		
		ASSERT( n2->findFinfo( "childSrc" )->add(
								n2, n22, n22->findFinfo( "child" ) ),
						"adding child"
			  );

		ASSERT( n1->connSize() == 2, "count children" );

		// n2 has n1 as parent, and n21 and n22 as children
		ASSERT( n2->connSize() == 3, "count children" );

		// Send the command to mark selected children for deletion.
		// In this case the selected child should be n2.
		sendTo1< int >( n1, 0, 0, 0 );

		// At this point n1 still has both n2 and n3 as children
		ASSERT( n1->connSize() == 2, "Should still have 2 children" );
		// and n2 still has n1 as parent, and n21 and n22 as children
		ASSERT( n2->connSize() == 3, "2 kids and a parent" );

		// Send the command to clean up messages. This still does
		// not delete anything.
		sendTo1< int >( n1, 0, 0, 1 );
		ASSERT( n1->connSize() == 1, "As far as n1 is concerned, n2 is removed" );
		// n2 still has n1 as parent, and n21 and n22 as children
		ASSERT( n2->connSize() == 3, "2 kids and a parent" );


		int initialNumInstances = SimpleElement::numInstances;
		// Finally, tell n2 to die. We can't use messages
		// any more because the handle has gone off n1.
		set< int >( n2, n2->findFinfo( "child" ), 2 );
		// Now we've gotten rid of n2.
		ASSERT( n1->connSize() == 1, "Now only 1 child." );

		// Now check that n2, n21, and n22 are really gwan.

		ASSERT( initialNumInstances - SimpleElement::numInstances == 3,
						"Check that n2, n21 and n22 are gone" );

		//////////////////////////////////////////////////////////
		// Testing create
		//////////////////////////////////////////////////////////
		

		set< string, string >( n1, n1->findFinfo( "create" ), 
						"Neutral", "N2" );
		ASSERT( initialNumInstances - SimpleElement::numInstances == 2,
						"Check that N2 is made" );


		//
		// It would be nice to have a findChild function. But
		// what would it return? Element ptr? Would not be
		// good across nodes.
		//
		// Likewise findParent.
}
#endif
