/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2005 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include <iostream>
#include "header.h"

#ifdef DO_UNIT_TESTS
	extern void testBasecode();
	extern void testNeutral();
	extern void testShell();
	extern void testInterpol();
#endif

#ifdef USE_GENESIS_PARSER
	extern void makeGenesisParser( const string& s );
#endif

int main(int argc, char** argv)
{
#ifdef DO_UNIT_TESTS
	testBasecode();
	testNeutral();
	testShell();
	testInterpol();
#endif

#ifdef USE_GENESIS_PARSER
	string line = "";
	if ( argc > 1 ) {
		int len = strlen( argv[1] );
		if ( len > 3 && strcmp( argv[1] + len - 2, ".g" ) == 0 )
			line = "include";
		else if ( len > 4 && strcmp( argv[1] + len - 3, ".mu" ) == 0 )
			line = "include";
		for ( int i = 1; i < argc; i++ )
			line = line + " " + argv[ i ];
	}
	makeGenesisParser( line );
#endif
	cout << "done" << endl;
}
