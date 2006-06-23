/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2005 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/
#include <fstream>
#include "header.h"
#include "../builtins/String.h"
#include "../builtins/Int.h"
#include "TestField.h"
#include "TestMsg.h"
// #include "../genesis_parser/GenesisParser.h"
// #include "../genesis_parser/GenesisParserWrapper.h"
#ifdef DO_UNIT_TESTS
	extern void testBasecode();
	extern void testScheduling();
	extern void testTable();
#endif

int main(int argc, const char** argv)
{
	Cinfo::initialize();

	Element* shell = Cinfo::find("Shell")->
		create( "sli_shell", Element::root() );
	Element* sli = Cinfo::find("GenesisParser")->
		create( "sli", shell );

	sli->field( "shell" ).set( "/sli_shell" );
	shell->field( "parser" ).set( "/sli_shell/sli" );
	Field f = sli->field( "process" ) ;

	Element* sched = Cinfo::find("Sched")->
		create( "sched", Element::root() );
	Cinfo::find("ClockJob")->create( "cj", sched );
	shell->field( "isInteractive" ).set( "0" );
	
#ifdef DO_UNIT_TESTS
	testBasecode();
	testScheduling();
	testTable();
#endif
	if ( argc > 1 ) {
		string line = "";
		int len = strlen( argv[ 1 ] );
		if ( len > 3 && strcmp( argv[ 1 ] + len - 2, ".g" ) == 0 )
			line = "include";
		if ( len > 4 && strcmp( argv[ 1 ] + len - 3, ".mu" ) == 0 )
			line = "include";
		// string line = "include";
		for (int i = 1; i < argc; i++)
			line = line + " " + argv[ i ];

		sli->field( "parse" ).set( line );
	}

	// setField( sli->field( "process" ) );
	f.set( "" );

	// setField( f );
}
