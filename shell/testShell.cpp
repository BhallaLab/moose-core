/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
#include "Shell.h"

void testCreateDelete()
{
	
	Eref ser = Id().eref();
	Id testId = Id::nextId();
	// Need to get the id back so that I can delete it later.
	bool ret = SetGet4< string, Id, Id, string >::set( ser, "create", "Neutral", Id(), testId , "testCreate" );
	assert( ret );

	ret = SetGet1< Id >::set( ser, "delete", testId );
	assert( ret );

	cout << "." << flush;
}

/**
 * Tests creation of SharedMsg from Shell to itself.
 * Will need to put in a deleteMsg as the Shell is a permanent fixture,
 * and dangling messages are bad.
 */
void testShellSharedMsg()
{
	Eref sheller = Id().eref();
	Shell* shell = reinterpret_cast< Shell* >( sheller.data() );

	bool ret = shell->doCreateMsg( Id(), "master", 
		Id(), "worker", "OneToOneMsg" );
	assert( ret );
	// sheller.element()->showMsg();
	cout << "." << flush;
}

/**
 * Tests Create and Delete calls issued through the parser interface,
 * which internally sets up blocking messaging calls.
 */
void testShellParserCreateDelete()
{
	Eref sheller = Id().eref();
	Shell* shell = reinterpret_cast< Shell* >( sheller.data() );

	bool ret = shell->doCreateMsg( Id(), "master", 
		Id(), "worker", "OneToOneMsg" );
	assert( ret );

//	sheller.element()->showFields();
//	sheller.element()->showMsg();

	vector< unsigned int > dimensions;
	Id child = shell->doCreate( "Neutral", Id(), "test", dimensions );

	shell->doDelete( child );
	cout << "." << flush;
}

/**
 * Tests Shell operations carried out on multiple nodes
 */
void testInterNodeOps()
{
	;
}

void testShell( )
{
	testCreateDelete();
	// testShellSharedMsg();
	testShellParserCreateDelete();
	//testInterNodeOps();
}
