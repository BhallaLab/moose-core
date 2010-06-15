/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
#include "ReadKkit.h"

void testReadKkit()
{
	ReadKkit rk;
	// rk.read( "test.g", "dend", 0 );
	rk.read( "dend_v26.g", "dend", 0 );

	cout << "." << flush;
}


void testKinetics()
{
	testReadKkit();
}

void testMpiKinetics( )
{
	//Need to update
// 	testMpiFibonacci();
}
