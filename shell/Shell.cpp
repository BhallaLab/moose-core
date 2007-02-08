/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2006 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include <map>
#include <algorithm>
#include "header.h"
#include "Cinfo.h"

using namespace std;

#include "DerivedFtype.h"
#include "Ftype2.h"
#include "setget.h"
#include "Shell.h"
#include "DynamicFinfo.h"
#include "MsgSrc.h" 
#include "MsgDest.h" 
#include "SimpleElement.h" 
#include "send.h"
#include "LookupFinfo.h"
#include "LookupFtype.h"
#include "setgetLookup.h"

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
unsigned int Shell::parent( unsigned int eid ) const 
{
	Element* e = Element::element( eid );
	unsigned int ret;
	// Check if eid is on local node, otherwise go to remote node
	
	if ( get< unsigned int >( e, "parent", ret ) )
		return ret;
	return 0;
}

/**
 * Returns the element at the end of the specified path
 * This function should be rethought because it puts parsing in the
 * shell rather than the parser, where it belongs.
 * We ignore any leading /
 * We ignore any isolated ./
 */
unsigned int Shell::traversePath(
				unsigned int start,
				vector< string >& names ) const
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
					return ret;
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

string Shell::eid2path( unsigned int eid ) const
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

void Shell::pwe() const
{
	cout << cwe_ << endl;
}

void Shell::ce( unsigned int dest )
{
	if ( Element::element( dest ) )
		cwe_ = dest;
}

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
