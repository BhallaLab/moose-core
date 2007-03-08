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
#include "Shell.h"

const unsigned int Shell::BAD_ID = ~0;

//////////////////////////////////////////////////////////////////////
// Shell MOOSE object creation stuff
//////////////////////////////////////////////////////////////////////

const Cinfo* initShellCinfo()
{
	/**
	 * This is a shared message to talk to the GenesisParser and
	 * perhaps to other parsers like the one for SWIG and Python
	 */

	static TypeFuncPair parserTypes[] =
	{
		// Setting cwe
		TypeFuncPair( Ftype1< unsigned int >::global(),
						RFCAST( &Shell::setCwe ) ),
		// Getting cwe back: First handle a request
		TypeFuncPair( Ftype0::global(), 
						RFCAST( &Shell::trigCwe ) ),
		// Then send out the cwe info
		TypeFuncPair( Ftype1< unsigned int >::global(), 0 ),

		// Getting a list of child ids: First handle a request with
		// the requested parent elm id.
		TypeFuncPair( Ftype1< unsigned int >::global(), 
						RFCAST( &Shell::trigLe ) ),
		// Then send out the vector of child ids.
		TypeFuncPair( Ftype1< vector< unsigned int > >::global(), 0 ),
		
		// Creating an object
		TypeFuncPair( 
				Ftype3< string, string, unsigned int >::global(),
				RFCAST( &Shell::staticCreate ) ),
		// The create func returns the id of the created object.
		TypeFuncPair( Ftype1< unsigned int >::global(), 0 ),
		// Deleting an object
		TypeFuncPair( 
				Ftype1< unsigned int >::global(), 
				RFCAST( &Shell::staticDestroy ) ),

		// Getting a field value as a string: handling request
		TypeFuncPair( 
				Ftype2< unsigned int, string >::global(),
				RFCAST( &Shell::getField ) ),
		// Getting a field value as a string: Sending value back.
		TypeFuncPair( Ftype1< string >::global(), 0 ),

		// Setting a field value as a string: handling request
		TypeFuncPair( 
				Ftype3< unsigned int, string, string >::global(),
				RFCAST( &Shell::setField ) ),
	};


	static Finfo* shellFinfos[] =
	{
		new ValueFinfo( "cwe", ValueFtype1< unsigned int >::global(),
				reinterpret_cast< GetFunc >( &Shell::getCwe ),
				RFCAST( &Shell::setCwe ) ),
		new SharedFinfo( "parser", parserTypes, 
				sizeof( parserTypes ) / sizeof( TypeFuncPair ) ), 
		new DestFinfo( "create",
				Ftype3< string, string, unsigned int >::global(),
				RFCAST( &Shell::staticCreate ) ),
		new DestFinfo( "destroy",
				Ftype1< unsigned int >::global(), 
				RFCAST( &Shell::staticDestroy ) ),
	};

	static Cinfo shellCinfo(
		"Shell",
		"Upi Bhalla, NCBS",
		"Shell object. Manages general simulator commands.",
		initNeutralCinfo(),
		shellFinfos,
		sizeof( shellFinfos ) / sizeof( Finfo* ),
		ValueFtype1< Shell >::global()
	);

	return &shellCinfo;
}

static const Cinfo* shellCinfo = initShellCinfo();
static const unsigned int cweSlot =
	initShellCinfo()->getSlotIndex( "parser" ) + 0;
static const unsigned int leSlot =
	initShellCinfo()->getSlotIndex( "parser" ) + 1;

// Returns the id of the created object
static const unsigned int createSlot =
	initShellCinfo()->getSlotIndex( "parser" ) + 2;
static const unsigned int getFieldSlot =
	initShellCinfo()->getSlotIndex( "parser" ) + 3;


//////////////////////////////////////////////////////////////////////
// Initializer
//////////////////////////////////////////////////////////////////////

Shell::Shell()
	: cwe_( 0 )
{;}

//////////////////////////////////////////////////////////////////////
// General path to eid conversion utilities
//////////////////////////////////////////////////////////////////////

/**
 * This needs to be on the Shell for future use, because we may
 * need to look up remote nodes
 */
unsigned int Shell::parent( unsigned int eid )
{
	Element* e = Element::element( eid );
	unsigned int ret;
	// Check if eid is on local node, otherwise go to remote node
	
	if ( get< unsigned int >( e, "parent", ret ) )
		return ret;
	return 0;
}

/**
 * Returns the id of the element at the end of the specified path.
 * On failure, returns Shell::BAD_ID
 * It is a static func as a utility for parsers.
 * It takes a pre-separated vector of names.
 * It ignores names that are just . or /
 */
unsigned int Shell::traversePath(
				unsigned int start,
				vector< string >& names )
{
	vector< string >::iterator i;
	for ( i = names.begin(); i != names.end(); i++ ) {
		if ( *i == "." || *i == "/" ) {
			continue;
		} else if ( *i == ".." ) {
			start = parent( start );
		} else {
			int ret;
			Element* e = Element::element( start );
			lookupGet< int, string >( e, "lookupChild", ret, *i );
			if ( ret == 0 )
					return BAD_ID;
			start = ret;
		}
	}
	return start;
}

/**
 * Chops up a string s into bits at separator, stuffs the bits
 * into the vector v.
 */
void separateString( const string& s, vector< string>& v, 
				const string& separator )
{
	string temp = s;
	unsigned int separatorLength = separator.length();
	string::size_type pos = s.find( separator );
	v.resize( 0 );

	while ( pos != string::npos ) {
		string t = temp.substr( 0, pos );
		if ( t.length() > 0 )
			v.push_back( t );
		temp = temp.substr( pos + separatorLength );
		pos = temp.find( separator );
	}
	if ( temp.length() > 0 )
		v.push_back( temp );
}

// Requires a path argument without a starting space
// Perhaps this should be in the interpreter?
unsigned int Shell::path2eid( 
		const string& path, const string& separator ) const
{
	if ( path == separator || path == "/root" )
			return 0;

	if ( path == "" || path == "." )
			return cwe_;

	if ( path == ".." ) {
			if ( cwe_ == 0 )
				return 0;
			return parent( cwe_ );
	}

	vector< string > names;

	unsigned int start;
	if ( path.substr( 0, separator.length() ) == separator ) {
		start = 0;
		separateString( path.substr( separator.length() ), names, separator );
	} else if ( path.substr( 0, 5 ) == "/root" ) {
		start = 0;
		separateString( path.substr( 5 ), names, separator );
	} else {
		start = cwe_;
		separateString( path, names, separator );
	}
	return traversePath( start, names );
}

string Shell::eid2path( unsigned int eid ) 
{
	static const string slash = "/";
	string n( "" );
	while ( eid != 0 ) {
		n = slash + Element::element( eid )->name() + n;
		eid = parent( eid );
	}

	/*
	Element* e = Element::element( eid );
	string name = "";

	while ( e != Element::root() ) {
		name = slash + e->name() + name;
		e = Element::element( parent( e->id() ) );
	}
	*/
	return n;
}

/**
 * Returns that component of path that precedes the last separator.
 * If there is nothing there, or no separator, returns an empty string.
 */
string Shell::head( const string& path, const string& separator )
{
	string::size_type pos = path.rfind( separator );
	if ( pos == string::npos )
			return "";

	return path.substr( 0, pos );
}

/**
 * Returns that component of path that follows the last separator.
 * If there is nothing there, or no separator, returns the entire path.
 */
string Shell::tail( const string& path, const string& separator )
{
	string::size_type pos = path.rfind( separator );
	if ( pos == string::npos )
			return path;

	return path.substr( pos + separator.length() );
}

//////////////////////////////////////////////////////////////////////
// Moose fields for Shell
//////////////////////////////////////////////////////////////////////

void Shell::setCwe( const Conn& c, unsigned int id )
{
	if ( id < Element::numElements() && Element::element( id ) != 0 ) {
		Shell* s = static_cast< Shell* >( c.targetElement()->data() );
		s->cwe_ = id;
	}
}

unsigned int Shell::getCwe( const Element* e )
{
	assert( e != 0 );
	const Shell* s = static_cast< const Shell* >( e->data() );
	return s->cwe_;
}

void Shell::trigCwe( const Conn& c )
						
{
	Shell* s = static_cast< Shell* >( c.targetElement()->data() );
	sendTo1< unsigned int >( c.targetElement(), cweSlot,
					c.targetIndex(), s->cwe_ );
}


//////////////////////////////////////////////////////////////////////
// Create and destroy are possibly soon to be deleted. These may have
// to go over to the Neutral, but till we've sorted out the SWIG
// interface we'll keep it in case it gets used there.
//////////////////////////////////////////////////////////////////////


void Shell::trigLe( const Conn& c, unsigned int parent )
						
{
	Element* pa = Element::element( parent );
	// Here we do something intelligent for off-node le.
	if ( pa ) {
		vector< unsigned int > ret;
		if ( get< vector< unsigned int > >( pa, "childList", ret ) ) {
			Element* e = c.targetElement();
			sendTo1< vector< unsigned int > >( e,
				leSlot, c.targetIndex(), ret );
		}
	}
}

// Static function
void Shell::staticCreate( const Conn& c, string type,
						string name, unsigned int parent )
{
	Shell* s = static_cast< Shell* >( c.targetElement()->data() );
	unsigned int ret = s->create( type, name, parent );
	if ( ret ) {
		sendTo1< unsigned int >( c.targetElement(),
					createSlot, c.targetIndex(), ret );
	}
}

// Static function
void Shell::staticDestroy( const Conn& c, unsigned int victim )
{
	Shell* s = static_cast< Shell* >( c.targetElement()->data() );
	s->destroy( victim );
}

// Static function
/**
 * This function handles request to get a field value. It triggers
 * a return function to the calling object, as a string.
 * The reason why we take this function to the Shell at all is because
 * we will eventually need to be able to handle this for off-node
 * object requests.
 */
void Shell::getField( const Conn& c, unsigned int id, string field )
{
	// Shell* s = static_cast< Shell* >( c.targetElement()->data() );
	string ret;
	Element* e = Element::element( id );
	// Appropriate off-node stuff here.

	const Finfo* f = e->findFinfo( field );
	if ( f ) {
		if ( f->strGet( e, ret ) )
			sendTo1< string >( c.targetElement(),
				getFieldSlot, c.targetIndex(), ret );
		else
			cout << "Error: Unable to get field " << e->name() <<
					"." << field << endl;
	} else {
		cout << "Error: field does not exist: " << e->name() <<
				"." << field << endl;
	}
}

// Static function
/**
 * This function handles request to set a field value.
 * The reason why we take this function to the Shell is because
 * we will eventually need to be able to handle this for off-node
 * object requests.
 */
void Shell::setField( const Conn& c, 
				unsigned int id, string field, string value )
{
	Element* e = Element::element( id );
	// Appropriate off-node stuff here.

	const Finfo* f = e->findFinfo( field );
	if ( f ) {
		if ( !f->strSet( e, value ) )
			cout << "Error: cannot set field " << e->name() <<
					"." << field << " to " << value << endl;
	} else {
		cout << "Error: cannot find field: " << e->name() <<
				"." << field << endl;
	}
}


// Regular function
unsigned int Shell::create( const string& type, const string& name, unsigned int parent )
{
	const Cinfo* c = Cinfo::find( type );
	Element* p = Element::element( parent );
	if ( !p ) {
		cout << "Error: Shell::create: No parent " << p << endl;
		return 0;
	}

	const Finfo* childSrc = p->findFinfo( "childSrc" );
	if ( !childSrc ) {
		// Sorry, couldn't resist it.
		cout << "Error: Shell::create: parent cannot handle child\n";
		return 0;
	}
	if ( c != 0 && p != 0 ) {
		Element* e = c->create( name );
		assert( childSrc->add( p, e, e->findFinfo( "child" ) ) );
		// cout << "OK\n";
		return e->id();
	} else  {
		cout << "Error: Shell::create: Unable to find type " <<
			type << endl;
	}
	return 0;
}

// Regular function
void Shell::destroy( unsigned int victim )
{
	// cout << "in Shell::destroy\n";
	Element* e = Element::element( victim );
	if ( !e ) {
		cout << "Error: Shell::destroy: No element " << victim << endl;
		return;
	}

	set( e, "destroy" );
}

//////////////////////////////////////////////////////////////////////
// Deleted stuff.
//////////////////////////////////////////////////////////////////////

#ifdef OLD_SHELL_FUNCS
void Shell::pwe() const
{
	cout << cwe_ << endl;
}

void Shell::ce( unsigned int dest )
{
	if ( Element::element( dest ) )
		cwe_ = dest;
}

void Shell::le ( unsigned int eid )
{
	Element* e = Element::element( eid );
	if ( e ) {
		vector< unsigned int > elist;
		vector< unsigned int >::iterator i;
		get( e, "childList", elist );
		for ( i = elist.begin(); i != elist.end(); i++ ) {
			if ( Element::element( *i ) != 0 )
				cout << Element::element( *i )->name() << endl;
		}
	}
}
#endif

#ifdef DO_UNIT_TESTS

#include "../element/Neutral.h"

void testShell()
{
	cout << "\nTesting Shell";

	Element* root = Element::root();
	ASSERT( root->id() == 0 , "creating /root" );

	vector< string > vs;
	separateString( "a/b/c/d/e/f/ghij/k", vs, "/" );

	/////////////////////////////////////////
	// Test path parsing
	/////////////////////////////////////////

	ASSERT( vs.size() == 8, "separate string" );
	ASSERT( vs[0] == "a", "separate string" );
	ASSERT( vs[1] == "b", "separate string" );
	ASSERT( vs[2] == "c", "separate string" );
	ASSERT( vs[3] == "d", "separate string" );
	ASSERT( vs[4] == "e", "separate string" );
	ASSERT( vs[5] == "f", "separate string" );
	ASSERT( vs[6] == "ghij", "separate string" );
	ASSERT( vs[7] == "k", "separate string" );

	separateString( "a->b->ghij->k", vs, "->" );
	ASSERT( vs.size() == 4, "separate string" );
	ASSERT( vs[0] == "a", "separate string" );
	ASSERT( vs[1] == "b", "separate string" );
	ASSERT( vs[2] == "ghij", "separate string" );
	ASSERT( vs[3] == "k", "separate string" );
	
	Shell sh;

	/////////////////////////////////////////
	// Test element creation in trees
	// This used to be a set of unit tests for Shell, but now
	// the operations have been shifted over to Neutral.
	// I still set up the creation operations because they are
	// used later for path lookup
	/////////////////////////////////////////

	unsigned int n = Element::numElements() - 1;

	unsigned int a = sh.create( "Neutral", "a", 0 );
	ASSERT( a == n + 1 , "creating a" );

	ASSERT( ( sh.parent( a ) == 0 ), "finding parent" );

	unsigned int b = sh.create( "Neutral", "b", 0 );
	ASSERT( b == n + 2 , "creating b" );

	unsigned int c = sh.create( "Neutral", "c", 0 );
	ASSERT( c == n + 3 , "creating c" );

	unsigned int a1 = sh.create( "Neutral", "a1", a );
	ASSERT( a1 == n + 4 , "creating a1" );

	ASSERT( ( sh.parent( a1 ) == a ), "finding parent" );

	unsigned int a2 = sh.create( "Neutral", "a2", a );
	ASSERT( a2 == n + 5 , "creating a2" );

	/////////////////////////////////////////
	// Test path lookup operations
	/////////////////////////////////////////

	string path = sh.eid2path( a1 );
	ASSERT( path == "/a/a1", "a1 eid2path" );
	path = sh.eid2path( a2 );
	ASSERT( path == "/a/a2", "a2 eid2path" );

	unsigned int eid = sh.path2eid( "/a/a1", "/" );
	ASSERT( eid == a1, "a1 path2eid" );
	eid = sh.path2eid( "/a/a2", "/" );
	ASSERT( eid == a2, "a2 path2eid" );

	/////////////////////////////////////////
	// Test destroy operation
	/////////////////////////////////////////
	sh.destroy( a );
	ASSERT( Element::element( a ) == 0, "destroy a" );
	ASSERT( Element::element( a1 ) == 0, "destroy a1" );
	ASSERT( Element::element( a2 ) == 0, "destroy a2" );
}

#endif // DO_UNIT_TESTS
