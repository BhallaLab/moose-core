/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "moose.h"
#include "Neutral.h"


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
	static Element* ret = initNeutralCinfo()->create( "root" );
	return ret;
}

/**
 * Declaration of the neutralCinfo() function is here because
 * we ensure the correct sequence of static initialization by having
 * each Cinfo use this call to find its base class. Most Cinfos
 * inherit from neutralCinfo. This function
 * uses the common trick of having an internal static value which
 * is created the first time the function is called.
 * The function for neutralCinfo has an additional line to statically
 * initialize the root element.
 */
const Cinfo* initNeutralCinfo()
{
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
		new ValueFinfo( "class", ValueFtype1< string >::global(),
					reinterpret_cast< GetFunc >( &Neutral::getClass ),
					&dummyFunc
		),
		new ValueFinfo( "childList",
				ValueFtype1< vector< unsigned int > >::global(), 
				reinterpret_cast< GetFunc>( &Neutral::getChildList ),
				&dummyFunc
		),
		/// Reports the cost of one clock tick, very roughly # of FLOPs.
		new ValueFinfo( "cpu",
				ValueFtype1< double >::global(), 
				reinterpret_cast< GetFunc>( &Neutral::getCpu ),
				&dummyFunc
		),
		/// Returns memory used by data part of object
		new ValueFinfo( "dataMem",
				ValueFtype1< unsigned int >::global(), 
				reinterpret_cast< GetFunc>( &Neutral::getDataMem ),
				&dummyFunc
		),
		/// Returns memory used by messaging (Element) part of object.
		new ValueFinfo( "msgMem",
				ValueFtype1< unsigned int >::global(), 
				reinterpret_cast< GetFunc>( &Neutral::getMsgMem ),
				&dummyFunc
		),
		/*
		/// Get and set the node on which this object sits. Low level op.
		new ValueFinfo( "node",
				ValueFtype1< unsigned int >::global(), 
				GFCAST( &Neutral::getNode ),
				RFCAST( &Neutral::setNode )
		),
		*/
		new LookupFinfo(
				"lookupChild",
				LookupFtype< unsigned int, string >::global(), 
				reinterpret_cast< GetFunc >( &Neutral::getChildByName ),
				0
		),
		new ValueFinfo( "fieldList",
				ValueFtype1< vector< string > >::global(), 
				reinterpret_cast< GetFunc>( &Neutral::getFieldList ),
				&dummyFunc
		),
		new SrcFinfo( "childSrc", Ftype1< int >::global() ),
		new DestFinfo( "child", Ftype1< int >::global(),
			reinterpret_cast< RecvFunc >( &Neutral::childFunc ) ),
		new DestFinfo( "create", Ftype2< string, string >::global(),
			reinterpret_cast< RecvFunc >( &Neutral::mcreate ) ),
		new DestFinfo( "destroy", Ftype0::global(),
			&Neutral::destroy ),

		/**
		 * This function allows objects to do additional initialization
		 * at the MOOSE level after they are created. For example, we
		 * may want to nest some fields in the object, which requires
		 * the creation of child objects to hold the nested fields.
		 * This would be done by overriding the postCreate function.
		 */
		new DestFinfo( "postCreate", Ftype0::global(), &dummyFunc ),
	};

	static Cinfo neutralCinfo(
				"Neutral",
				"Upi Bhalla",
				"Neutral object. Manages Element heirarchy.",
				0,
				neutralFinfos,
				sizeof( neutralFinfos ) / sizeof( Finfo* ),
				ValueFtype1< Neutral >::global()
	);

	return &neutralCinfo;
}

static const Cinfo* neutralCinfo = initNeutralCinfo();
static const Element* root = Element::root();
const unsigned int Neutral::childSrcIndex = initNeutralCinfo()->
	getSlotIndex( "childSrc" );
const unsigned int Neutral::childIndex = initNeutralCinfo()->
	getSlotIndex( "child" );


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
					///\todo: Need to cleanly delete the data part too.
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

const string Neutral::getClass( const Element* e )
{
		return e->className();
}

/////////////////////////////////////////////////////////////////////
// Field functions.
/////////////////////////////////////////////////////////////////////

// Perhaps this should take a Cinfo* for the first arg, except that
// I don't want to add yet another class into the header.
// An alternative would be to use an indexed lookup for all Cinfos
void Neutral::mcreate( const Conn& conn,
				const string cinfo, const string name )
{
		create( cinfo, name, conn.targetElement() );
/*
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
*/
}

/**
 * Underlying utility function for creating objects.
 */
Element* Neutral::create(
		const string& cinfo, const string& name, Element* parent )
{
	// Need to check here if the name is an existing one.
	const Cinfo* c = Cinfo::find( cinfo );
	if ( c ) {
		Element* kid = c->create( name );
		// Here a global absolute or a relative finfo lookup for
		// the childSrc field would be useful.
		parent->findFinfo( "childSrc" )->
				add( parent, kid, kid->findFinfo( "child" ) ); 
		return kid;
	} else {
		cout << "Error: Neutral::create: class " << cinfo << 
				" not found\n";
	}
	return 0;
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
unsigned int Neutral::getChildByName( const Element* elm, const string& s )
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
	// Failure option: return BAD_ID.
	return BAD_ID;
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

vector< unsigned int > Neutral::getChildList( const Element* e )
{
	// const SimpleElement* e = dynamic_cast< const SimpleElement *>(elm);
	// assert( e != 0 );

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

vector< string > Neutral::getFieldList( const Element* elm )
{
	// const SimpleElement* e = dynamic_cast< const SimpleElement *>(elm);
	// assert( e != 0 );

	vector< string > ret;
	vector< const Finfo* > flist;
	vector< const Finfo* >::const_iterator i;
	elm->listFinfos( flist );

	for ( i = flist.begin(); i != flist.end(); i++ )
		ret.push_back( (*i)->name() );

	return ret;
}

double Neutral::getCpu( const Element* e )
{
	return 0.0;
}

unsigned int Neutral::getDataMem( const Element* e )
{
	const Finfo *f = e->getThisFinfo();
	return f->ftype()->size();
}

unsigned int Neutral::getMsgMem( const Element* e )
{
	return e->getMsgMem();
}

/*
unsigned int Neutral::getNode( const Element* e )
{
	return e->getNode();
}

void Neutral::setNode( const Conn& c, unsigned int node )
{
	e->setNode( node );
}
*/

/////////////////////////////////////////////////////////////////////
// Unit tests.
/////////////////////////////////////////////////////////////////////

#ifdef DO_UNIT_TESTS
#include <algorithm>
#include "Ftype2.h"
#include "setget.h"

void testNeutral()
{
		cout << "\nTesting Neutral";

		Element* n1 = neutralCinfo->create( "n1" );
		string s;
		get< string >( n1, n1->findFinfo( "name" ), s );
		ASSERT( s == "n1", "Neutral name get" );
		set< string >( n1, n1->findFinfo( "name" ), "N1" );
		s = "";
		get< string >( n1, n1->findFinfo( "name" ), s );
		ASSERT( s == "N1", "Neutral name set" );

		Element* n2 = neutralCinfo->create( "n2" );
		
		ASSERT( n1->findFinfo( "childSrc" )->add(
								n1, n2, n2->findFinfo( "child" ) ),
						"adding child"
			  );

		Element* n3 = neutralCinfo->create( "n3" );
		
		ASSERT( n1->findFinfo( "childSrc" )->add(
								n1, n3, n3->findFinfo( "child" ) ),
						"adding child"
			  );

		Element* n21 = neutralCinfo->create( "n21" );
		
		ASSERT( n2->findFinfo( "childSrc" )->add(
								n2, n21, n21->findFinfo( "child" ) ),
						"adding child"
			  );

		Element* n22 = neutralCinfo->create( "n22" );
		
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

		Element* foo = Neutral::create( "Neutral", "foo", n1 );
		ASSERT( foo != 0, "Neutral::create" );
		ASSERT( initialNumInstances - SimpleElement::numInstances == 1,
						"Check that foo is made" );
		ASSERT( foo->name() == "foo", "Neutral::create" );

		//
		// It would be nice to have a findChild function. But
		// what would it return? Element ptr? Would not be
		// good across nodes.
		//
		// Likewise findParent.
}
#endif
