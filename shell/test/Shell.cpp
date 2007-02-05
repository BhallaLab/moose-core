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
#include <iostream>
#include "Swig.h"
#include "Shell.h"

using namespace std;

/////////////////////////////////////////////////////////////////////

static Shell* sh()
{
		static Shell* ret = new Shell();

		return ret;
}

/////////////////////////////////////////////////////////////////////

const string& pwe()
{
		return sh()->pwe();
}

void ce( const string& dest )
{
			sh()->ce( dest );
}

void create( const string& path )
{
			sh()->create( path );
}

void remove( const string& path )
{
			sh()->remove( path );
}

void le ( const string& dest ) 
{
			sh()->le( dest );
}

void le ( )
{
			sh()->le( );
}

/////////////////////////////////////////////////////////////////////

Shell::Shell()
	: cwe_("/")
{;}

const string& Shell::pwe() const
{
	return cwe_;
}


string Shell::expandPath( const string& path ) const
{
	static string root = "/";
	if ( path == "." )
			return cwe_;

	if ( path == ".." ) {
			if ( cwe_ == "/" )
				return cwe_;
			if ( cwe_.length() < 2 ) {
				return root;
			}
			string::size_type pos = cwe_.rfind( "/" );
			if ( pos == string::npos ) {
					cout << "Bad cwe = " << cwe_ << endl;
					return root;
			}
			return cwe_.substr( 0, pos );
	} else {
			return path;
	}
}

void Shell::ce( const string& dest )
{
	cwe_ = expandPath( dest );
	cout << "OK\n";
}

void Shell::create( const string& path )
{
		cout << "in Shell::create\n";
}

void Shell::remove( const string& path )
{
		cout << "in Shell::remove\n";
}

void Shell::le ( const string& path )
{
		cout << "in Shell::le( path )\n";
};


void Shell::le ( )
{
		cout << "in Shell::le()\n";
};
