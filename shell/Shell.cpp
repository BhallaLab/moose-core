/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2006 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include <string>
#include <map>
#include <iostream>
#include "header.h"
#include "Cinfo.h"

using namespace std;

#include "DerivedFtype.h"
#include "Ftype2.h"
#include "setget.h"
#include "Shell.h"

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
 * We ignore any leading /
 * We ignore any isolated ./
 */
unsigned int Shell::traversePath(
				unsigned int start, const string& s ) const
{
	if ( s.length() == 0 )
			return start;

	if ( s[0] == '/' )
			return traversePath( start, s.substr( 1 ) );

	if ( s[0] == '.' ) {
		if ( s.length() == 1 )
			return start;
		if ( s[1] == '/' )
			return traversePath( start, s.substr( 2 ) );
		if ( s[1] == '.' )
			return traversePath( parent( start ), s.substr( 2 ) );

		// If none of these, do the usual string comparison with name.
		// Here we get all the child eids, and scan for match.
	}
	return 0;
}

// Requires a path argument without a starting space
// Perhaps this should be in the interpreter?
unsigned int Shell::path2eid( const string& path ) const
{
	if ( path == "/" || path == "/root" )
			return 0;

	if ( path == "" || path == "." )
			return cwe_;

	if ( path == ".." ) {
			if ( cwe_ == 0 )
				return 0;
			return parent( cwe_ );
	}

	unsigned int eid;
	if ( path[0] == '/' )
		eid = traversePath( 0, path.substr( 1 ) );
	else if ( path.substr( 0, 5 ) == "/root" )
		eid = traversePath( 0, path.substr( 5 ) );
	else if ( path.substr( 0, 2 ) == "./" )
		eid = traversePath( cwe_, path.substr( 2 ) );
	else if ( path.substr( 0, 3 ) == "../" )
		eid = traversePath( parent( cwe_ ), path.substr( 3 ) );

	return eid;
}

string Shell::eid2path( unsigned int eid ) const
{
	static string slash = "/";
	Element* e = Element::element( eid );
	string name = "";

	while ( e != Element::root() ) {
		name = slash + e->name() + name;
	}
	return name;
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

void Shell::create( const string& type, const string& name, unsigned int parent )
{
	cout << "in Shell::create\n";
	const Cinfo* c = Cinfo::find( type );
	Element* p = Element::element( parent );
	if ( !p ) {
		cout << "Error: Shell::create: No parent " << p << endl;
		return;
	}

	const Finfo* childSrc = p->findFinfo( "childSrc" );
	if ( !childSrc ) {
		// Sorry, couldn't resist it.
		cout << "Error: Shell::create: parent cannot handle child\n";
		return;
	}
	if ( c != 0 && p != 0 ) {
		Element* e = c->create( name );
		assert( childSrc->add( p, e, e->findFinfo( "child" ) ) );
		cout << "OK\n";
	} else  {
		cout << "Error: Shell::create: Unable to find type " <<
			type << endl;
	}
}

void Shell::destroy( unsigned int victim )
{
	cout << "in Shell::destroy\n";
	Element* e = Element::element( victim );
	if ( !e ) {
		cout << "Error: Shell::destroy: No element " << victim << endl;
		return;
	}

	set( e, "destroy" );
}

void Shell::le ( unsigned int eid )
{
		cout << "in Shell::le( path )\n";
}
