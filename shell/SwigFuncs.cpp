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

/////////////////////////////////////////////////////////////////////

static Shell* sh()
{
		static Shell* ret = new Shell();

		return ret;
}

/////////////////////////////////////////////////////////////////////

void pwe()
{
		return sh()->pwe();
}

void ce( const string& dest )
{
		sh()->ce( sh()->path2eid( dest ) );
}

void create( const string& type, const string& path )
{
	string::size_type pos = path.rfind( "/" );
	string name;
	if ( pos == string::npos ) {
		sh()->create( type, path, sh()->cwe() );
	} else {
		sh()->create( type, path.substr( pos ), 
						sh()->path2eid( path.substr( 0, pos - 1 ) ) );
	}
}

void destroy( const string& path )
{
		sh()->destroy( sh()->path2eid( path ) );
}

void le ( const string& path ) 
{
	sh()->le( sh()->path2eid( path ) );
}

void le ( )
{
	sh()->le( sh()->cwe() );
}
