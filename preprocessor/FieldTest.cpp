#include <sstream>
#include "Field.h"

FieldTest( vector< int >& error )
{
	ostringstream output;

	Field f1( "double", "t1", RW, 0, 0 );
	f1.printH( output );
	if ( output.str() != "	double t1;\n")
		error.push_back( 1 );
	/*
	f1.printWrapperH( output );
	if ( output.str() != "	double t1;\n")
		error.push_back( 1 );
		*/

	FieldInfo f2( "long", "t2", CONST, 0, 0 );
	f2.setConst();
	f2.print_h( output );
	if ( output.str() != "	const long t2;\n")
		error.push_back( 1 );
};
