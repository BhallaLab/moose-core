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
#include "shell/Shell.h"


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
	static Element* ret = initNeutralCinfo()->create( Id(), "root" );
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
                                reinterpret_cast< RecvFunc >( &Neutral::setName ),
                                "Name of the object."
                                
		),
		new ValueFinfo( "index", ValueFtype1< int >::global(),
                                reinterpret_cast< GetFunc >( &Neutral::getIndex ),
                                &dummyFunc,
                                "Index of the object if it is an array element."
		),
		new ValueFinfo( "parent", ValueFtype1< Id >::global(),
                                reinterpret_cast< GetFunc >( &Neutral::getParent ),
                                &dummyFunc,
                                "Parent object of this object."
		),
		new ValueFinfo( "class", ValueFtype1< string >::global(),
                                reinterpret_cast< GetFunc >( &Neutral::getClass ),
                                &dummyFunc,
                                "Class of this object."
		),
		new ValueFinfo( "childList",
				ValueFtype1< vector< Id > >::global(), 
				reinterpret_cast< GetFunc>( &Neutral::getChildList ),
				&dummyFunc,
                                "List of children of this object."
		),
		new ValueFinfo( "node",
				ValueFtype1< unsigned int >::global(),
				reinterpret_cast< GetFunc>( &Neutral::getNode ),
				&dummyFunc,
                                "CPU Node in which this object resides."
		),
		new ValueFinfo( "cpu",
				ValueFtype1< double >::global(), 
				reinterpret_cast< GetFunc>( &Neutral::getCpu ),
				&dummyFunc,
				"Reports the cost of one clock tick, very roughly # of FLOPs."
		),
		new ValueFinfo( "dataMem",
				ValueFtype1< unsigned int >::global(), 
				reinterpret_cast< GetFunc>( &Neutral::getDataMem ),
				&dummyFunc,
				"Returns memory used by data part of object"
		),
		new ValueFinfo( "msgMem",
				ValueFtype1< unsigned int >::global(), 
				reinterpret_cast< GetFunc>( &Neutral::getMsgMem ),
				&dummyFunc,
				"Returns memory used by messaging (Element) part of object."
		),
		new LookupFinfo(
				"lookupChild",
				LookupFtype< Id, string >::global(), 
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
		new DestFinfo( "createArray", Ftype3< string, string, int >::global(),
			reinterpret_cast< RecvFunc >( &Neutral::mcreateArray ) ),
		new DestFinfo( "destroy", Ftype0::global(),
			&Neutral::destroy ),

		new DestFinfo( "postCreate", Ftype0::global(), &dummyFunc, 
			"This function allows objects to do additional initialization at the MOOSE level after they are created. "
			"For example, we may want to nest some fields in the object, which requires the creation of child "
			"objects to hold the nested fields. This would be done by overriding the postCreate function." ),
	};
	
	static string doc[] =
	{
		"Name", "Neutral",
		"Author", "Upi Bhalla",
		"Description", "Neutral object. Manages Element heirarchy.",
	};
	static Cinfo neutralCinfo(
				doc,
				sizeof( doc ) / sizeof( string ),				
				0,
				neutralFinfos,
				sizeof( neutralFinfos ) / sizeof( Finfo* ),
				ValueFtype1< Neutral >::global()
	);

	return &neutralCinfo;
}

static const Cinfo* neutralCinfo = initNeutralCinfo();
// static const Element* root = Element::root();
// const Slot Neutral::childIndex = initNeutralCinfo()->
//	getSlotIndex( "child" );

static const Slot childSrcSlot = initNeutralCinfo()->
	getSlot( "childSrc" );

//////////////////////////////////////////////////////////////////
// Here we put the Neutral class functions.
//////////////////////////////////////////////////////////////////

/**
 * This function is called to recursively delete all children
 * It is a bit tricky, because while we delete things the conn
 * vector gets altered as each child is removed. So the iterators
 * don't work.
 * It actually needs to work in three stages.
 * 1. Mark all children for deletion.
 * 2. Clear out messages outside local set, without altering
 * local Conn arrays.
 * 3. Delete.
 *
 * There is a further complication for arrays. We don't permit removal
 * of individual entries: the whole array goes. Later we may permit
 * shrinking the array: the last entry can be removed. For now we
 * do the operation only for the index 0.
 * Also need to work out what to do for index AnyIndex
 */
void Neutral::childFunc( const Conn* c , int stage )
{
		Element* e = c->target().e;
		assert( stage == 0 || stage == 1 || stage == 2 );
		if ( c->target().i > 0 )
			return;

		switch ( stage ) {
				case MARK_FOR_DELETION:
					send1< int >( e, childSrcSlot, MARK_FOR_DELETION );
					e->prepareForDeletion( 0 );
				break;
				case CLEAR_MESSAGES:
					send1< int >( e, childSrcSlot, CLEAR_MESSAGES );
					e->prepareForDeletion( 1 );
				break;
				case COMPLETE_DELETION:
					send1< int >( e, childSrcSlot, COMPLETE_DELETION );
					///\todo: Need to cleanly delete the data part too.
					delete e;
				break;
				default:
					assert( 0 );
				break;
		}
}

const string Neutral::getName( Eref e )
{
		return e.saneName( getParent( e ));
}

void Neutral::setName( const Conn* c, const string s )
{
	c->target().e->setName( s );
}

const int Neutral::getIndex( Eref e )
{
		return e.i;
}

const string Neutral::getClass( Eref e )
{
		return e.e->className();
}

/////////////////////////////////////////////////////////////////////
// Field functions.
/////////////////////////////////////////////////////////////////////

// Perhaps this should take a Cinfo* for the first arg, except that
// I don't want to add yet another class into the header.
// An alternative would be to use an indexed lookup for all Cinfos
void Neutral::mcreate( const Conn* conn,
				const string cinfo, const string name )
{
	create( cinfo, name, conn->target().id(), Id::scratchId() );
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

void Neutral::mcreateArray( const Conn* conn,
				const string cinfo, const string name, int n )
{
		createArray( cinfo, name, conn->target().id(), Id::scratchId(), n );
}


/**
 * Underlying utility function for creating objects.
 */
Element* Neutral::create(
		const string& cinfo, const string& name, Id parent, Id id )
{
	// Check that the parent exists.
	if ( parent.bad() ) {
		cout << "Error: Neutral::create: No parent\n";
		return 0;
	}
	// Check that the parent can handle children
	const Finfo* childSrc = parent()->findFinfo( "childSrc" );
	if ( !childSrc ) {
		cout << "Error: Neutral::create: object '" << parent()->name() << 
			"' cannot handle child\n";
		return 0;
	}

	// Check that the child class is correct
	const Cinfo* c = Cinfo::find( cinfo );
	if ( !c ) {
		cout << "Error: Neutral::create: class " << cinfo << 
				" not found\n";
		return 0;
	}

	// Need to check here if the name is an existing one.
	Id existing = getChildByName( parent.eref(), name );
	if ( existing.good() ) {
		cout << "Error: Neutral::create: Attempt to overwrite existing element '" << existing.path() << "'. Using original.\n";
		return existing();
	}

	if ( c ) {
		// Element* kid = c->create( Id::scratchId(), name );
		Element* kid = c->create( id, name );
		const Finfo* kFinfo = kid->findFinfo( "child" );
		assert( kFinfo != 0 );
		// Here a global absolute or a relative finfo lookup for
		// the childSrc field would be useful.
		bool ret = parent.eref().add( childSrc->msg(), kid, kFinfo->msg(),
			ConnTainer::Default );
		// bool ret = childSrc->add( parent, kid, kFinfo );
		assert( ret );
		ret = c->schedule( kid, ConnTainer::Default );
		assert( ret );
		return kid;
	}
	return 0;
}

Element* Neutral::createArray(
		const string& cinfo, const string& name, Id parent, Id id, int n )
{
	// Check that the parent exists.
	if ( parent.bad() ) {
		cout << "Error: Neutral::create: No parent\n";
		return 0;
	}
	// Check that the parent can handle children
	const Finfo* childSrc = parent()->findFinfo( "childSrc" );
	if ( !childSrc ) {
		cout << "Error: Neutral::create: object '" << parent()->name() << 
			"' cannot handle child\n";
		return 0;
	}

	// Check that the child class is correct
	const Cinfo* c = Cinfo::find( cinfo );
	if ( !c ) {
		cout << "Error: Neutral::create: class " << cinfo << 
				" not found\n";
		return 0;
	}

	// Need to check here if the name is an existing one.
	Id existing = getChildByName( parent.eref(), name );
	if ( existing.good() ) {
		cout << "Error: Neutral::create: Attempt to overwrite existing element '" << existing.path() << "'. Using original.\n";
		return existing();
	}

	if ( c ) {
		// Element* kid = c->create( Id::scratchId(), name );
		Element* kid = c->createArray( id, name, n );
		const Finfo* kFinfo = kid->findFinfo( "child" );
		assert( kFinfo != 0 );
		// Here a global absolute or a relative finfo lookup for
		// the childSrc field would be useful.
		// bool ret = childSrc->add( parent, kid, kFinfo );
		bool ret = parent.eref().add( childSrc->msg(), kid, kFinfo->msg(),
			ConnTainer::One2All );
			// childSrc->add( parent, kid, kFinfo );
		assert( ret );
		ret = c->schedule( kid, ConnTainer::One2All );
		assert( ret );
		return kid;
	}
	return 0;
	
	/*// Need to check here if the name is an existing one.
	const Cinfo* c = Cinfo::find( cinfo );
	if ( c ) {
		Element* kid = c->createArray( Id::scratchId(), name, n, 0 );
		// Here a global absolute or a relative finfo lookup for
		// the childSrc field would be useful.
		bool ret = parent->findFinfo( "childSrc" )->
				add( parent, kid, kid->findFinfo( "child" ) ); 
		assert( ret );
		ret = c->schedule( kid );
		assert( ret );
		return kid;
	} else {
		cout << "Error: Neutral::create: class " << cinfo << 
				" not found\n";
	}
	return 0;*/
}


void Neutral::destroy( const Conn* c )
{
	childFunc( c, MARK_FOR_DELETION );
	childFunc( c, CLEAR_MESSAGES );
	childFunc( c, COMPLETE_DELETION );
}

Id Neutral::getParent( Eref e )
{
	assert( e.e != 0 );
	if ( e.e == Element::root() )
		return Id();
        Conn* c = e.e->targets( "child", e.i );
	assert( c->good() );
	Id parent = c->target().id();
	delete c;
        return parent;        
}

string str(int a){
	char e[20];
	sprintf(e, "%d", a);
	return string(e);
}

/**
 * Looks up the child with the specified name, and returns its id.
 */
Id Neutral::getChildByName( Eref er, const string& s )
{
	Element *e = er.e;
	assert( e != 0 );
	assert( s.length() > 0 );
	const Msg* m = e->msg( childSrcSlot.msg() );
	assert( m != 0 );
	vector< ConnTainer* >::const_iterator i;
	
	string name;
	unsigned int index = 0;
	if ( s[s.length() - 1] == ']' ) {
		string::size_type pos = s.rfind( '[' );
		if ( pos == string::npos )
			return Id::badId();
		if ( pos == s.length() - 2 ) {
			// return the whole array
			name = s.substr( 0, pos );
			index = 0;
		} else {
			name = s.substr( 0, pos );
			index = atoi( s.substr( pos + 1, s.length() - pos ).c_str() );
		}
	} else {
		name = s;
		index = er.i;
	}

	do {
		for ( i = m->begin(); i != m->end(); i++ ) {
			// Going through ConnTainers here, not Conns
			Element* e2 = ( *i )->e2();
			//takes care of simple elements which are of the form cc[2]
			if ( e2->name() == s && e2->elementType() == "Simple")
				return e2->id();
	
			if ( e2->name() == name ){
				if ( e2->elementType() == "Simple" && index == 0 )
					return e2->id();
				else if ( e2->elementType() == "Array" )
					return e2->id().assignIndex( index );
				// Here we put in an option for proxies that handle
				// off-node children. This assumes that e2->name() returns the
				// child name and not some dummy string from the proxy.
				// We will also need options for array proxies. Those get 
				// messy because they may be scattered over many nodes
				else if ( e2->elementType() == "Proxy" )
					return e2->id();
			}
		}
		/*
		if ( ( ( *i )->e2()->name() == name + '[' + str(index) + ']' ) && ( *i )->e2()->elementType() == "Simple")
			return ( *i )->e2()->id().assignIndex( index );
		*/
		m = m->next( e );
	} while ( m );
	return Id::badId();
}

/**
 * Looks up the child with the specified name, and sends its eid
 * in a message back to sender.
 */
void Neutral::lookupChild( const Conn* c, const string s )
{
	Id ret = getChildByName( c->target(), s );
	sendBack1< Id >( c, childSrcSlot, ret );
}

vector< Id > Neutral::getChildList( Eref e )
{
	vector< Id > ret;
	getChildren( e, ret );
	return ret;
}

void Neutral::getChildren( const Eref e, vector< Id >& ret )
{
	assert( e.e != 0 );
	ret.resize( 0 );
	Conn* c = e.e->targets( childSrcSlot.msg(), e.i );
	while ( c->good() ) {
		ret.push_back( c->target().id() );
		c->increment();
	}
	delete c;
}

vector< string > Neutral::getFieldList( Eref elm )
{
	// const SimpleElement* e = dynamic_cast< const SimpleElement *>(elm);
	// assert( e != 0 );

	vector< string > ret;
	vector< const Finfo* > flist;
	vector< const Finfo* >::const_iterator i;
	elm.e->listFinfos( flist );

	for ( i = flist.begin(); i != flist.end(); i++ )
		ret.push_back( (*i)->name() );

	return ret;
}

unsigned int Neutral::getNode( Eref e )
{
	return e->id().node();
}

double Neutral::getCpu( Eref e )
{
	return 0.0;
}

unsigned int Neutral::getDataMem( Eref e )
{
	const Finfo *f = e.e->getThisFinfo();
	return f->ftype()->size();
}

unsigned int Neutral::getMsgMem( Eref e )
{
	return e.e->getMsgMem();
}

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
		const Finfo* childSrcFinfo = 
			Element::root()->findFinfo( "childSrc" );
		assert( childSrcFinfo != 0 );
		int childDestMsg = Element::root()->findFinfo( "child" )->msg();

		Element* n1 = neutralCinfo->create( Id::scratchId(), "N1" );
		bool ret = Eref::root().add( "childSrc", n1, "child" );
		// bool ret = childSrcFinfo->add( Element::root(), n1, n1->findFinfo( "child" ) );
		ASSERT( ret, "adding n1");

		string s;
		get< string >( n1, n1->findFinfo( "name" ), s );
		ASSERT( s == "N1", "Neutral name get" );
		set< string >( n1, n1->findFinfo( "name" ), "n1" );
		s = "";
		get< string >( n1, n1->findFinfo( "name" ), s );
		ASSERT( s == "n1", "Neutral name set" );

		Element* n2 = neutralCinfo->create( Id::scratchId(), "n2" );
		
		ret = Eref( n1 ).add( "childSrc", n2, "child" );
		// ret = childSrcFinfo->add( n1, n2, n2->findFinfo( "child" ) );
		ASSERT( ret , "adding child");

		Element* n3 = neutralCinfo->create( Id::scratchId(), "n3" );
		
		ret = Eref( n1 ).add( "childSrc", n3, "child" );
		// ret = childSrcFinfo->add( n1, n3, n3->findFinfo( "child" ) );
		ASSERT( ret, "adding child");

		Element* n21 = neutralCinfo->create( Id::scratchId(), "n21" );
		
		ret = Eref( n2 ).add( "childSrc", n21, "child" );
		// ret = childSrcFinfo->add( n2, n21, n21->findFinfo( "child" ) );
		ASSERT( ret, "adding child");

		Element* n22 = neutralCinfo->create( Id::scratchId(), "n22" );
		
		ret = Eref( n2 ).add( "childSrc", n22, "child" );
		// ret = childSrcFinfo->add( n2, n22, n22->findFinfo( "child" ) );
		ASSERT( ret, "adding child");

		ASSERT( n1->msg( childSrcSlot.msg() )->size() == 2, "count children and parent" );
		ASSERT( n1->dest( childDestMsg )->size() == 1,
			"count children and parent" );

		// n2 has n1 as parent, and n21 and n22 as children
		ASSERT( n2->msg( childSrcSlot.msg() )->size() == 2, "count children" );
		ASSERT( n2->dest( childDestMsg )->size() == 1, "count parent" );

		// Send the command to mark selected children for deletion.
		// In this case the selected child should be n2.
		sendTo1< int >( n1, childSrcSlot, 0, 0 );

		// At this point n1 still has both n2 and n3 as children
		ASSERT( n1->msg( childSrcSlot.msg() )->size() == 2, "Should still have 2 children and parent" );
		// and n2 still has n1 as parent, and n21 and n22 as children
		ASSERT( n2->msg( childSrcSlot.msg() )->size() == 2, "2 kids and a parent" );

		// Send the command to clean up messages. This still does
		// not delete anything, but now n2 and children are isolated.
		// The CLEAR_MESSAGES (== 1) flag says to delete all messages 
		// outside delete tree, which includes n1 here because n1 is 
		// outside delete tree.
		// But n3 should still be there as a child of n1.
		sendTo1< int >( n1, childSrcSlot, 0, 1 );
		ASSERT( n1->msg( childSrcSlot.msg() )->size() == 1,
			"Now n1 should have only n3 as a child." );
		ASSERT( ( *n1->msg( childSrcSlot.msg() )->begin() )->e2() == n3, 
			"n3 is the only remaining child." );
		// n2 still has n1 as parent, and n21 and n22 as children
		ASSERT( n2->msg( childSrcSlot.msg() )->size() == 2,
			"2 kids and a parent" );
		ASSERT( n2->dest( childDestMsg )->size() == 0, "2 kids, no parent" );


		int initialNumInstances = SimpleElement::numInstances;
		// Finally, tell n2 to die. We can't use messages
		// any more because the handle has gone off n1.
		// sendTo1< int >( n1, 0, childSrcSlot, 0, 2 );
		set< int >( n2, "child", 2 );
		// Now we've gotten rid of n2.
		ASSERT( n1->msg( childSrcSlot.msg() )->size() == 1, "Now only 1 child." );

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

		Element* foo = Neutral::create( "Neutral", "foo", n1->id(), 
			Id::scratchId() );
		ASSERT( foo != 0, "Neutral::create" );
		ASSERT( initialNumInstances - SimpleElement::numInstances == 1,
						"Check that foo is made" );
		ASSERT( foo->name() == "foo", "Neutral::create" );
		ret = set( n1, "destroy" );
		ASSERT( ret, "cleaning up n1" );

		//
		// It would be nice to have a findChild function. But
		// what would it return? Element ptr? Would not be
		// good across nodes.
		//
		// Likewise findParent.
}
#endif
