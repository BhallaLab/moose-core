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
#include "../shell/Shell.h"

void testKsolveZombify()
{
	ReadKkit rk;
	// rk.read( "test.g", "dend", 0 );
	Id base = rk.read( "foo.g", "dend", Id() );
	assert( base != Id() );
	// Id kinetics = s->doFind( "/kinetics" );

	Shell* s = reinterpret_cast< Shell* >( Id().eref().data() );
	vector< unsigned int > dims( 1, 1 );
	Id stoich = s->doCreate( "Stoich", base, "stoich", dims );
	assert( stoich != Id() );
	string temp = "/dend/##";
	bool ret = Field<string>::set( stoich.eref(), "path", temp );
	assert( ret );

	/*
	rk.run();
	rk.dumpPlots( "dend.plot" );
	*/

	s->doDelete( base );
	cout << "." << flush;
}

void testKsolve()
{
	testKsolveZombify();
}

void testMpiKsolve()
{
}
