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
#include "Swig.h"
using namespace std;

#include "header.h"
#include "Cinfo.h"
#include "Shell.h"

static string separator = "/";

/////////////////////////////////////////////////////////////////////

static Shell* sh()
{
		static Shell* ret = new Shell();

		return ret;
}

/////////////////////////////////////////////////////////////////////

/*
 * This has to change to use the shell as a MOOSE object, not a
 * direct pointer.
 */

void pwe()
{
		return;
}

void ce( const string& dest )
{
	// sh()->ce( sh()->path2eid( dest, separator ) );
	return;
}

void create( const string& type, const string& path )
{
	string::size_type pos = path.rfind( separator );
	string name;
	if ( pos == string::npos ) {
		// sh()->create( type, path, sh()->cwe() );
	} else if ( pos == 0 ) {
		sh()->create( type, path.substr( separator.length() ), 0 );
	} else {
		string head = path.substr( 0, pos );
		string tail = path.substr( pos + separator.length() );
		sh()->create( type, tail, sh()->path2eid( head, separator, 0 ) );
		cout << "creating " << type << " on " << 
			head << "(" << sh()->path2eid( head, separator, 0 ) <<
			") named " << tail << endl;
	}
}

void destroy( const string& path )
{
		sh()->destroy( sh()->path2eid( path, separator, 0 ) );
}

void le ( const string& path ) 
{
	// sh()->le( sh()->path2eid( path, separator ) );
	return;
}

void le ( )
{
	// sh()->le( sh()->cwe() );
	return;
}
