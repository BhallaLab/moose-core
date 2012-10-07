/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2005 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/
#ifdef DO_UNIT_TESTS

#include "header.h"
#include "Adaptor.h"

void testAdaptor()
{
	Adaptor foo;
	foo.setInputOffset( 1 );
	foo.setOutputOffset( 2 );
	foo.setScale( 10 );

	for ( unsigned int i = 0; i < 10; ++i )
		foo.input( i );


	assert( doubleEq( foo.getOutput(), 0.0 ) );
	foo.innerProcess();

	assert( doubleEq( foo.getOutput(), ( -1.0 + 4.5) * 10.0 + 2.0 ) );

	// shell->doDelete( nid );
	cout << "." << flush;
}

// This tests stuff without using the messaging.
void testSigNeur()
{
	testAdaptor();
}

// This is applicable to tests that use the messaging and scheduling.
void testSigNeurProcess()
{
}

#endif
