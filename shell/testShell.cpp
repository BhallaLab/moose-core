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

void testCreate()
{
	
	Eref ser = Id().eref();
	// Need to get the id back so that I can delete it later.
	bool ret = SetGet4< string, Id, Id, string >::set( ser, "create", "Neutral", Id(), Id(2) , "testCreate" );
	assert( ret );
	cout << "." << flush;
}

void testShell( )
{
	testCreate();
}
